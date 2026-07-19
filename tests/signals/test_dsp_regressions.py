"""Physics-based regression tests for corrected signal-processing behavior."""

from unittest.mock import patch

import numpy as np
import pytest

from torchsig.signals.builders.chirpss import chirpss_modulator_baseband
from torchsig.transforms.functional import coarse_gain_change, digital_agc, iq_imbalance
from torchsig.transforms.transforms import DigitalAGC


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
