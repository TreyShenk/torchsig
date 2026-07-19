"""Unit tests for physical impairments and opt-in ML augmentations."""

from torchsig.signals.signal_types import Signal
from torchsig.transforms.base_transforms import RandAugment, Transform
from torchsig.transforms.impairments import Impairments, MLAugmentations
from torchsig.utils.dsp import compute_spectrogram

import numpy as np
import pytest


@pytest.mark.parametrize(
    "params, is_error",
    [
        ({"level": 0}, False),
        ({"level": 1}, False),
        ({"level": 2}, False),
        ({"level": 42}, True),
    ],
)
def test_Impairments(params: dict, is_error: bool) -> None:
    """Test Impairments with pytest.

    Args:
        params (dict): Parameter specifying impairment level.
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test output.

    """
    level = params["level"]

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            T = Impairments(level=level, seed=42)
    else:
        T = Impairments(level=level, seed=42)

        assert isinstance(T, Impairments)
        assert isinstance(T.level, int)
        assert isinstance(T.random_generator, np.random.Generator)
        for t in T.signal_transforms.transforms:
            assert isinstance(t, Transform)
        for t in T.dataset_transforms.transforms:
            assert isinstance(t, Transform)


def test_level_zero_impairments_preserve_flat_noise_spectrum() -> None:
    """Level zero must not apply the former random AddSlope augmentation."""
    rng = np.random.default_rng(12345)
    data = (rng.standard_normal(262144) + 1j * rng.standard_normal(262144)).astype(np.complex64)
    signal = Signal(data=data.copy())

    impairments = Impairments(level=0, seed=0)
    assert impairments.dataset_transforms.transforms == []
    output = impairments.dataset_transforms(signal)

    assert np.array_equal(output.data, data)
    spectrum_db = compute_spectrogram(output.data, fft_size=512, fft_stride=512).mean(axis=1)
    dc_power_db = np.mean(spectrum_db[255:257])
    edge_power_db = np.mean(np.concatenate((spectrum_db[:16], spectrum_db[-16:])))
    assert abs(edge_power_db - dc_power_db) < 1.0


def test_ml_augmentations_are_explicit_and_seeded() -> None:
    """The former level-zero augmentations remain available only by opt-in."""
    augmentation_a = MLAugmentations(seed=42)
    augmentation_b = MLAugmentations(seed=42)

    assert isinstance(augmentation_a.transforms[0], RandAugment)
    assert {type(transform).__name__ for transform in augmentation_a.transforms[0].transforms} == {
        "AddSlope",
        "ChannelSwap",
        "RandomDropSamples",
        "TimeReversal",
    }

    data = (np.arange(512, dtype=np.float32) + 1j * np.arange(512, dtype=np.float32)[::-1]).astype(np.complex64)
    output_a = augmentation_a(Signal(data=data.copy()))
    output_b = augmentation_b(Signal(data=data.copy()))

    assert np.array_equal(output_a.data, output_b.data)
    assert not np.array_equal(output_a.data, data)
