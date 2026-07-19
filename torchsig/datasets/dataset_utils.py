"""Dataset Utilities"""

import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import (
    frequency_shift,
    upconversion_anti_aliasing_filter,
)

# name of yaml file where dataset information will be written
dataset_yaml_name = "create_dataset_info.yaml"
# name of yaml file where dataset writing information will be written
writer_yaml_name = "writer_info.yaml"


def frequency_shift_signal(
    signal: Signal,
    center_freq_min: float,
    center_freq_max: float,
    sample_rate: float,
    frequency_max: float,
    frequency_min: float,
    random_generator: np.random.Generator | None = None,
) -> Signal:
    """Randomly shifts a signal while keeping its occupied bandwidth in bounds.

    Args:
        signal (Signal): The signal object to be frequency shifted.
        center_freq_min (float): Minimum requested center frequency for the random shift.
        center_freq_max (float): Maximum requested center frequency for the random shift.
        sample_rate (float): The sample rate of the signal.
        frequency_max (float): Maximum permitted occupied frequency.
        frequency_min (float): Minimum permitted occupied frequency.
        random_generator (np.random.Generator, optional): Random number generator for generating the random shift. Defaults to `np.random.default_rng()`.

    Returns:
        Signal: The frequency-shifted signal with updated metadata.

    """
    random_generator = np.random.default_rng(seed=None) if random_generator is None else random_generator

    # Keep the clean occupied bandwidth inside the configured frequency window.
    # This avoids silently reshaping ordinary generated signals with the
    # anti-aliasing filter after their bandwidth has been measured.
    center_freq_lower = max(center_freq_min, frequency_min + signal.bandwidth / 2)
    center_freq_upper = min(center_freq_max, frequency_max - signal.bandwidth / 2)
    if center_freq_lower <= center_freq_upper:
        # Randomize within the bandwidth-valid center-frequency interval.
        center_freq = random_generator.uniform(low=center_freq_lower, high=center_freq_upper)
    else:
        # Some intentionally extreme configurations cannot fit. Retain the
        # established anti-aliasing path below for those cases rather than
        # rejecting the whole generated sample.
        center_freq = random_generator.uniform(low=center_freq_min, high=center_freq_max)

    # frequency shift to center_freq
    signal.data = frequency_shift(signal.data, center_freq, sample_rate)

    # update center_freq field in metadata
    signal["center_freq"] = center_freq

    # calculate upper and lower frequency edges of signal
    upper_freq = signal.upper_freq
    lower_freq = signal.lower_freq

    # This is a numerical-safety fallback and handles configurations whose
    # requested signal bandwidth cannot fit in the configured frequency window.
    if upper_freq > frequency_max or lower_freq < frequency_min:
        # apply an anti-aliasing filter to the signal to attenuate energy that
        # wrapped around -fs/2 or fs/2. additionally, due to the filtering the
        # bandwidth changed bandwidth, and therefore changed the center frequency,
        # so update the two metadata fields accordingly
        signal.data, signal["center_freq"], signal["bandwidth"] = (
            upconversion_anti_aliasing_filter(
                signal.data,
                signal["center_freq"],
                signal["bandwidth"],
                sample_rate,
                frequency_max,
                frequency_min,
            )
        )
    # do nothing

    # center frequency is now set, and therefore can be verified
    signal["center_freq_set"] = True

    return signal


def save_type(transforms: list, target_transforms: list):
    """Determines if the dataset will generate 'raw' IQ data, which means no transform and target transforms have been applied.

    Args:
        transforms (list): A list of transformations to be applied to the data.
        target_transforms (list): A list of target transformations.

    Returns:
        bool: `True` if no transformations are applied, indicating raw IQ data; otherwise `False`.
    """
    return len(transforms) == 0 and len(target_transforms) == 0
