"""Regression tests for signal-generation defaults and family aliases."""

from unittest.mock import patch

import numpy as np
import pytest

from torchsig.datasets.dataset_utils import frequency_shift_signal
from torchsig.signals.builder import ConcatSignalGenerator
from torchsig.signals.builders.am import am_modulator
from torchsig.signals.builders.constellation import (
    ConstellationSignalGenerator,
    constellation_modulator_baseband,
)
from torchsig.signals.builders.fm import FMSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.utils.signal_building import lookup_signal_generator_by_string


def test_am_dsb_uses_a_nonnegative_real_envelope() -> None:
    signal = am_modulator(
        am_mode="dsb",
        bandwidth=100_000.0,
        sample_rate=1_000_000.0,
        num_samples=16_384,
        rng=np.random.default_rng(0),
    )

    np.testing.assert_allclose(signal.imag, 0.0, atol=1e-6)
    assert signal.real.min() >= -1e-6


def test_dataset_constellation_generator_uses_srrc_pulse_shape() -> None:
    generator = ConstellationSignalGenerator(
        metadata={
            "constellation_name": "qpsk",
            "sample_rate": 1_000_000.0,
            "bandwidth_min": 100_000.0,
            "bandwidth_max": 100_000.0,
            "signal_duration_in_samples_min": 256,
            "signal_duration_in_samples_max": 256,
        },
        seed=0,
    )

    with patch(
        "torchsig.signals.builders.constellation.constellation_modulator",
        return_value=np.ones(256, dtype=np.complex64),
    ) as modulator_mock:
        signal = generator()

    args = modulator_mock.call_args.args
    assert args[1] == "srrc"
    assert 0.1 <= args[5] <= 0.5
    assert signal.pulse_shape_name == "srrc"
    assert signal.alpha_rolloff == args[5]


def test_rectangular_constellation_modulator_remains_explicitly_available() -> None:
    output = constellation_modulator_baseband(
        constellation_name="qpsk",
        pulse_shape_name="rectangular",
        max_num_samples=256,
        oversampling_rate_nominal=4,
        rng=np.random.default_rng(0),
    )

    assert output.shape == (256,)
    assert np.iscomplexobj(output)


def test_frequency_placement_keeps_clean_bandwidth_inside_limits() -> None:
    signal = Signal(
        data=np.ones(128, dtype=np.complex64),
        center_freq=0.0,
        bandwidth=3_000_000.0,
    )

    with patch("torchsig.datasets.dataset_utils.upconversion_anti_aliasing_filter") as filter_mock:
        output = frequency_shift_signal(
            signal,
            center_freq_min=-2_500_000.0,
            center_freq_max=2_500_000.0,
            sample_rate=10_000_000.0,
            frequency_min=-2_500_000.0,
            frequency_max=2_500_000.0,
            random_generator=np.random.default_rng(0),
        )

    assert output.lower_freq >= -2_500_000.0
    assert output.upper_freq <= 2_500_000.0
    filter_mock.assert_not_called()


def test_frequency_placement_filters_signal_that_cannot_fit() -> None:
    signal = Signal(
        data=np.ones(128, dtype=np.complex64),
        center_freq=0.0,
        bandwidth=5_000_001.0,
    )

    with patch(
        "torchsig.datasets.dataset_utils.upconversion_anti_aliasing_filter",
        return_value=(signal.data, 0.0, 5_000_000.0),
    ) as filter_mock:
        output = frequency_shift_signal(
            signal,
            center_freq_min=-2_500_000.0,
            center_freq_max=2_500_000.0,
            sample_rate=10_000_000.0,
            frequency_min=-2_500_000.0,
            frequency_max=2_500_000.0,
            random_generator=np.random.default_rng(0),
        )

    assert output.bandwidth == 5_000_000.0
    filter_mock.assert_called_once()


OFDM_NAMES = {
    "ofdm-64",
    "ofdm-72",
    "ofdm-128",
    "ofdm-180",
    "ofdm-256",
    "ofdm-300",
    "ofdm-512",
    "ofdm-600",
    "ofdm-900",
    "ofdm-1024",
    "ofdm-1200",
    "ofdm-2048",
}
AM_NAMES = {"am-dsb", "am-dsb-sc", "am-usb", "am-lsb"}
PSK_NAMES = {"bpsk", "qpsk", "8psk", "16psk", "32psk", "64psk"}
QAM_NAMES = {
    "16qam",
    "32qam",
    "64qam",
    "256qam",
    "1024qam",
    "32qam_cross",
    "128qam_cross",
    "512qam_cross",
}
ASK_NAMES = {"4ask", "8ask", "16ask", "32ask", "64ask"}
FSK_NAMES = {
    f"{constellation_size}{fsk_type}"
    for constellation_size in (2, 4, 8, 16)
    for fsk_type in ("fsk", "gfsk", "msk", "gmsk")
}
MSK_NAMES = {name for name in FSK_NAMES if name.endswith("msk")}
LFM_NAMES = {"lfm-data", "lfm-radar"}


@pytest.mark.parametrize(
    ("family_name", "expected_members"),
    [
        ("ofdm", OFDM_NAMES),
        ("am", AM_NAMES),
        ("fsk", FSK_NAMES),
        ("psk", PSK_NAMES),
        ("qam", QAM_NAMES),
        ("ask", ASK_NAMES),
        ("lfm", LFM_NAMES),
        ("msk", MSK_NAMES),
    ],
)
def test_family_aliases_have_explicit_memberships(
    family_name: str, expected_members: set[str]
) -> None:
    generator = lookup_signal_generator_by_string(family_name)

    assert isinstance(generator, ConcatSignalGenerator)
    assert {member.class_name for member in generator.signal_generators} == expected_members


def test_fm_alias_selects_standalone_analog_fm() -> None:
    generator = lookup_signal_generator_by_string("fm")

    assert isinstance(generator, FMSignalGenerator)
    assert generator.class_name == "fm"
