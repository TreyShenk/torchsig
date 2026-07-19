"""Physics-based regression tests for corrected signal-processing behavior."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.signals.builders.chirpss import chirpss_modulator_baseband
from torchsig.signals.builders.fsk import fsk_modulator
from torchsig.signals.signal_types import Signal
from torchsig.transforms.functional import coarse_gain_change, digital_agc, iq_imbalance
from torchsig.transforms.transforms import DigitalAGC
from torchsig.utils.dsp import (
    estimate_occupied_bandwidth,
    polyphase_decimator,
    update_signal_snr_bandwidth,
)


def test_coarse_gain_change_uses_amplitude_db() -> None:
    data = np.ones(16, dtype=np.complex64)

    output = coarse_gain_change(data, gain_change_db=20.0, start_idx=8)

    np.testing.assert_allclose(output[:8], 1.0)
    np.testing.assert_allclose(output[8:], 10.0)


def test_digital_agc_initial_gain_uses_db() -> None:
    data = np.ones(16, dtype=np.complex64)

    output = digital_agc(
        data,
        initial_gain_db=20.0,
        alpha_smooth=0.0,
        alpha_track=0.0,
        alpha_overflow=0.0,
        alpha_acquire=0.0,
        ref_level_db=0.0,
        track_range_db=1.0,
        low_level_db=-200.0,
        high_level_db=200.0,
    )

    np.testing.assert_allclose(np.abs(output), 10.0)


def test_digital_agc_uses_configured_overflow_rate() -> None:
    transform = DigitalAGC(
        alpha_track=(1e-5, 1e-5),
        alpha_overflow=(0.2, 0.2),
        seed=0,
    )

    assert transform.alpha_overflow_distribution() == pytest.approx(0.2)


def test_iq_amplitude_imbalance_sets_requested_gain_ratio() -> None:
    data = np.full(32, 1 + 1j, dtype=np.complex64)

    output = iq_imbalance(
        data,
        amplitude_imbalance=6.0,
        phase_imbalance=0.0,
        dc_offset_db=-300.0,
        dc_offset_phase_rads=0.0,
        noise_power_db=0.0,
    )

    measured_ratio_db = 20 * np.log10(np.mean(np.abs(output.real)) / np.mean(np.abs(output.imag)))
    assert measured_ratio_db == pytest.approx(6.0, abs=1e-5)


def test_chirpss_uses_bandwidth_as_total_sweep_width() -> None:
    bandwidth = 1 / 4

    with patch(
        "torchsig.signals.builders.chirpss.chirp",
        side_effect=lambda _f0, _f1, samples: np.ones(samples, dtype=np.complex64),
    ) as chirp_mock:
        chirpss_modulator_baseband(
            max_num_samples=256,
            oversampling_rate_nominal=4,
            rng=np.random.default_rng(0),
        )

    f0, f1, _ = chirp_mock.call_args.args
    assert f0 == pytest.approx(-bandwidth / 2)
    assert f1 == pytest.approx(bandwidth / 2)


def test_polyphase_decimator_preserves_dc_gain() -> None:
    data = np.ones(4096, dtype=np.complex64)

    output = polyphase_decimator(data, decimation_rate=2)

    np.testing.assert_allclose(output[128:-128], 1.0, atol=1e-6)


def test_fsk_resampling_does_not_apply_bandwidth_dependent_gain() -> None:
    resampled = np.ones(64, dtype=np.complex64)

    with patch(
        "torchsig.signals.builders.fsk.multistage_polyphase_resampler",
        return_value=resampled,
    ):
        output = fsk_modulator(
            constellation_size=2,
            fsk_type="fsk",
            bandwidth=100.0,
            sample_rate=800.0,
            num_samples=64,
            rng=np.random.default_rng(0),
        )

    np.testing.assert_allclose(output, 1.0)


def test_dataset_frequency_rectangle_uses_physical_signal_edges() -> None:
    dataset = TorchSigIterableDataset(
        signal_generators=[],
        validate_init=False,
        sample_rate=1000.0,
        fft_size=100,
    )
    signal = Signal(
        data=np.ones(200, dtype=np.complex64),
        center_freq=100.0,
        bandwidth=200.0,
    )

    rectangle = dataset._map_to_coordinates(signal, start_sample=0)  # noqa: SLF001

    assert rectangle.coord_lower_left.y == pytest.approx(50.0)
    assert rectangle.coord_upper_right.y == pytest.approx(70.0)


def test_upper_frequency_setter_keeps_positive_bandwidth() -> None:
    signal = Signal(
        data=np.ones(1, dtype=np.complex64), center_freq=100.0, bandwidth=20.0
    )
    _ = signal.lower_freq

    signal.upper_freq = 130.0

    assert signal.center_freq == pytest.approx(110.0)
    assert signal.bandwidth == pytest.approx(40.0)


def test_lower_frequency_setter_keeps_positive_bandwidth() -> None:
    signal = Signal(
        data=np.ones(1, dtype=np.complex64), center_freq=100.0, bandwidth=20.0
    )
    _ = signal.upper_freq

    signal.lower_freq = 70.0

    assert signal.center_freq == pytest.approx(90.0)
    assert signal.bandwidth == pytest.approx(40.0)


def test_occupied_bandwidth_uses_equal_power_tails() -> None:
    ascending_power = np.array(
        [0.004, 0.006, 0.49, 1e-12, 0.49, 0.006, 0.004, 1e-12]
    )
    spectrogram_db = 10 * np.log10(ascending_power[::-1, np.newaxis])

    bandwidth = estimate_occupied_bandwidth(
        spectrogram_db,
        sample_rate=1000.0,
    )

    assert bandwidth == pytest.approx(625.0)


def test_clean_occupied_bandwidth_is_independent_of_target_snr() -> None:
    sample_rate = 1024.0
    sample_index = np.arange(4096)
    clean_data = np.exp(2j * np.pi * 0.125 * sample_index).astype(np.complex64)
    dataset = SimpleNamespace(
        fft_size=256,
        fft_stride=256,
        sample_rate=sample_rate,
        noise_power_db=-100.0,
        random_generator=np.random.default_rng(0),
    )
    low_snr_signal = Signal(
        data=clean_data.copy(),
        snr_db_min=0.0,
        snr_db_max=0.0,
    )
    high_snr_signal = Signal(
        data=clean_data.copy(),
        snr_db_min=50.0,
        snr_db_max=50.0,
    )

    update_signal_snr_bandwidth(dataset, low_snr_signal)
    update_signal_snr_bandwidth(dataset, high_snr_signal)

    assert low_snr_signal.bandwidth == pytest.approx(high_snr_signal.bandwidth)
