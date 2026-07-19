from typing import Any

from torchsig.signals.builder import ConcatSignalGenerator
from torchsig.signals.builders.am import AMSignalGenerator
from torchsig.signals.builders.chirpss import ChirpSSSignalGenerator
from torchsig.signals.builders.constellation import ConstellationSignalGenerator
from torchsig.signals.builders.constellation_maps import all_symbol_maps
from torchsig.signals.builders.fm import FMSignalGenerator
from torchsig.signals.builders.fsk import FSKSignalGenerator
from torchsig.signals.builders.lfm import LFMSignalGenerator
from torchsig.signals.builders.ofdm import OFDMSignalGenerator
from torchsig.signals.builders.tone import ToneSignalGenerator

# Stores generator class and metadata for generators to make per label
signal_generator_lookup_table: dict[str,
    tuple[type, dict[str, any]] |
    tuple[type, list[tuple[type, dict[str, Any]]], dict[str, Any]]
] = {}

# Initialize lookup table with signal generators
signal_generator_lookup_table["tone"] = (ToneSignalGenerator, {})
num_subcarrier_values = [64, 72, 128, 180, 256, 300, 512, 600, 900, 1024, 1200, 2048]
ofdm_signal_names = ["ofdm-" + str(num_subcarriers) for num_subcarriers in num_subcarrier_values]
for num_subcarriers in num_subcarrier_values:
    signal_generator_lookup_table["ofdm-" + str(num_subcarriers)] = (
        OFDMSignalGenerator,
        {"num_subcarriers": num_subcarriers},
    )
lfm_signal_names = ["lfm-data", "lfm-radar"]
signal_generator_lookup_table["lfm-data"] = (LFMSignalGenerator, {"lfm_type": "data"})
signal_generator_lookup_table["lfm-radar"] = (LFMSignalGenerator, {"lfm_type": "radar"})
fsk_signal_names = []
for fsk_type in ["fsk", "gfsk", "msk", "gmsk"]:
    for constellation_size in [2, 4, 8, 16]:
        signal_name = str(constellation_size) + str(fsk_type)
        fsk_signal_names.append(signal_name)
        signal_generator_lookup_table[signal_name] = (
            FSKSignalGenerator,
            {"fsk_type": fsk_type, "constellation_size": constellation_size},
        )
signal_generator_lookup_table["fm"] = (FMSignalGenerator, {})
constellation_signal_names = list(all_symbol_maps)
for constellation_name in all_symbol_maps:
    signal_generator_lookup_table[constellation_name] = (
        ConstellationSignalGenerator,
        {"constellation_name": constellation_name},
    )
signal_generator_lookup_table["chirpss"] = (ChirpSSSignalGenerator, {})
am_signal_names = ["am-dsb", "am-dsb-sc", "am-usb", "am-lsb"]
for am_mode in ["dsb", "dsb-sc", "usb", "lsb"]:
    signal_generator_lookup_table["am-" + am_mode] = (
        AMSignalGenerator,
        {"am_mode": am_mode},
    )
signal_generator_lookup_table["all"] = (
    ConcatSignalGenerator,
    [
        signal_generator_lookup_table[key]
        for key in signal_generator_lookup_table
    ],
    {},
)
family_signal_names = {
    "ofdm": ofdm_signal_names,
    "am": am_signal_names,
    "fsk": fsk_signal_names,
    "psk": [name for name in constellation_signal_names if name.endswith("psk")],
    "qam": [name for name in constellation_signal_names if "qam" in name],
    "ask": [name for name in constellation_signal_names if name.endswith("ask")],
    "lfm": lfm_signal_names,
    "msk": [name for name in fsk_signal_names if name.endswith("msk")],
}
for family_name, signal_names in family_signal_names.items():
    signal_generator_lookup_table[family_name] = (
        ConcatSignalGenerator,
        [
            signal_generator_lookup_table[signal_name]
            for signal_name in signal_names
        ],
        {"family_name": family_name},
    )


def lookup_signal_generator_by_string(signal_generator_name: str) -> Any:
    """Look up and instantiate a signal generator by its name.

    This function searches the signal_generator_lookup_table for the given name
    and returns an instantiated signal generator. It handles both simple generators
    and concatenated generators (ConcatSignalGenerator).

    Args:
        signal_generator_name: The name of the signal generator to instantiate.

    Returns:
        An instantiated signal generator object.

    Raises:
        ValueError: If the signal generator name is not found in the lookup table
            or if there's an error instantiating the generator.
    """
    try:
        lookup_value = signal_generator_lookup_table[signal_generator_name]
        if len(lookup_value) == 2:
            generator_init, metadata = lookup_value
            return generator_init(metadata=metadata)
        if len(lookup_value) == 3 and lookup_value[0] == ConcatSignalGenerator:
            generator_init, generator_list, metadata = lookup_value
            return generator_init(
                signal_generators=[el[0](metadata=el[1]) for el in generator_list],
                metadata=metadata,
            )
        raise KeyError("bad data found in generator lookup table")
    except KeyError:
        raise ValueError(
            "could not instantiate signal generator: '"
            + str(signal_generator_name)
            + "'"
        )
