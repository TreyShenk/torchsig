<a align="center" href="https://torchsig.com">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="docs/torchsig_logo_white_dodgerblue.png">
        <img src="docs/logo.png" width="500">
    </picture>
</a>

-----

## Fork purpose and scope

This is an independently maintained fork of [TorchSig](https://torchsig.com) focused on physically consistent synthetic RF IQ and ground-truth metadata for local testing, detector evaluation, and development. It is not an official TorchSig release and does not submit changes or pull requests to the upstream project. Upstream TorchSig remains credited under its original license; this fork preserves its API where practical while correcting clear generation and labeling issues.

### Fork highlights

- **Clean occupied-bandwidth labels:** `bandwidth` is measured from each isolated, clean waveform using a 99% equal-tail occupied-power estimate before SNR scaling or dataset noise is added.
- **Corrected DSP calibration:** amplitude-dB conversion, AGC units, IQ imbalance, ChirpSS sweep width, resampler gain, FSK scaling, and frequency-edge metadata have targeted corrections and regression tests.
- **More realistic default linear modulation:** dataset-generated PSK/QAM/ASK uses SRRC pulse shaping and records its pulse shape and rolloff in metadata. Explicit rectangular modulation remains available for deliberate experiments.
- **Bandwidth-safe placement:** component center frequencies are sampled so their clean occupied bandwidth fits the configured frequency window, avoiding ordinary post-placement spectral clipping.
- **Unambiguous signal families:** `fm` means analog FM only; LFM, AM, QAM, PSK, FSK, MSK, and OFDM family aliases have explicit memberships.
- **Clean means clean:** impairment level 0 now produces unmodified IQ. Optional, nonphysical ML augmentations are opt-in through `MLAugmentations()`.

### Current limitations

Clock jitter/drift, FM bandwidth calibration, OFDM cyclic-prefix defaults, and the physical receiver-filter model behind `PassbandRipple` still need deliberate modeling decisions. Regenerate datasets after these corrections: previously generated data is not statistically equivalent to data from this fork.

See [STATUS.md](STATUS.md) for the technical rationale, validation results, and current priorities.

[TorchSig](https://torchsig.com) is an open-source signal processing machine learning toolkit based on the PyTorch data handling pipeline. The toolkit simplifies common digital signal processing operations, augmentations, and transformations when dealing with both real and complex-valued signals, particularly within (but not limited to) the radio-frequency domain.

# Getting Started

## Prerequisites
- Ubuntu &ge; 22.04
- Hard drive storage with &ge; 1 TB
- CPU with &ge; 4 cores
- GPU with &ge; 16 GB storage (recommended)
- Python &ge; 3.10

We highly reccomend Ubuntu or using a Docker container.

## Installation
Clone the `torchsig` repository and install using the following commands:
```
git clone https://github.com/TreyShenk/torchsig.git
cd torchsig
pip install -e .
```

To install this fork directly into another project, use:

```
pip install "git+https://github.com/TreyShenk/torchsig.git"
```

# Examples and Tutorials

TorchSig has a series of Jupyter notebooks in the `examples/` directory. View the README inside `examples/` to learn more.

# Usage

## Generating Datasets with Python
TorchSig uses a unified dataset architecture. Create datasets using the Python API:
```python
# Physically clean, label-consistent IQ. Level 0 does not add augmentations.
from torchsig.datasets.datasets import StaticTorchSigDataset
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import default_dataset
from torchsig.utils.writer import DatasetCreator

dataset = default_dataset(
    impairment_level=0,
    signal_generators=["ofdm"],
)

# Optional training-only augmentation. Do not use this for calibrated evaluation.
from torchsig.transforms.impairments import MLAugmentations

augmented_dataset = default_dataset(
    impairment_level=0,
    signal_generators=["ofdm"],
    transforms=[MLAugmentations()],
)

# Add physical receiver impairments only when the scenario requires them.
impaired_dataset = default_dataset(
    impairment_level=1,
    signal_generators=["ofdm"],
)

# Create a reproducible dataloader for the clean dataset.
dataloader = WorkerSeedingDataLoader(dataset, batch_size=2)

# Save the dataset to disk.
dataset_creator = DatasetCreator(
    dataset_length=20,
    dataloader=dataloader,
    root="./sample_dataset",
    overwrite=True,
    multithreading=False,
)
dataset_creator.create()

# Load it back from disk.
static_dataset = StaticTorchSigDataset(root="./sample_dataset")

print(static_dataset[0])
```

# Docker
One option for running TorchSig is within Docker. Start by building the Docker container:

```bash
docker build -t torchsig -f docker/Dockerfile .
```

And then you can launch a Docker instance:
```bash
docker run -it torchsig
```
See `docker/README.md` to learn more.

# Development
To contribute to our library, please make sure to run the following:

```bash
# pytests all pass
pytest

# pylint score > 9/10
pylint --rcfile=.pylintrc torchsig

# not required
# but helpful for maintaining PEP 8 Style Guide
ruff check torchsig
```
Both need to pass in order to contribute to our Github.

# Key Features
TorchSig provides many useful tools to facilitate and accelerate research on signals processing machine learning technologies:
- **Unified Dataset Architecture**: TorchSig features a single, flexible dataset system that supports both signal classification (single signal) and signal detection (multiple signals) tasks through configuration.
- **Comprehensive Signal Library**: Support for 60+ signal types across all major modulation families (FSK, QAM, PSK, ASK, OFDM, Analog) with realistic impairments and channel effects.
- **Advanced Transform System**: Numerous signals processing transforms enable existing ML techniques to be employed on signals data, with unified impairment models supporting perfect, cabled, and wireless channel conditions.

## Core Classes
- **`Signal` and `SignalMetadataObject`**: Enable signal objects and metadata to be seamlessly handled and operated on throughout the TorchSig infrastructure.
- **`TorchSigIterableDataset`**: Unified dataset class that synthetically creates, augments, and transforms signals datasets. Behavior (classification vs detection) is determined by configuration parameters.
  - Can generate samples infinitely when `num_samples=None`, or finite datasets when `num_samples` is specified.
  - Dataset type determined by `num_signals_max`: 1 for classification, >1 for detection tasks.
- **`DatasetCreator`**: Writes a PyTorch `DataLoader` containing a `TorchSigIterableDataset` objects to disk with progress tracking and memory optimization.
- **`StaticTorchSigDataset`**: Loads previously generated datasets from disk back into memory.
  - Can access previously generated samples efficiently.
  - Supports both classification and detection datasets through unified interface.



# Documentation
Documentation can be found [online](https://torchsig.readthedocs.io/latest/) or built locally by following the instructions below.
```
cd docs
pip install -r docs-requirements.txt
make html
firefox build/html/index.html
```


# License
TorchSig is released under the MIT License. The MIT license is a popular open-source software license enabling free use, redistribution, and modifications, even for commercial purposes, provided the license is included in all copies or substantial portions of the software. TorchSig has no connection to MIT, other than through the use of this license.

# Publications
| Title | Year  | Cite (APA) |
| ----- | ----  | ---------- |
| [TorchSig 2.0: Dataset Customization, New Transforms and Future Plans](https://events.gnuradio.org/event/26/contributions/752/) | 2025 | Oh, E., Mullins, J., Carrick, M., Vondal, M., Hoffman, J., Leonardo, F., Toliver, P., Miller, R. (2025, September). TorchSig 2.0: Dataset Customization, New Transforms and Future Plans. In Proceedings of the GNU Radio Conference (Vol. 10, No. 1). |
| [TorchSig: A GNU Radio Block and New Spectrogram Tools for Augmenting ML Training](https://events.gnuradio.org/event/24/contributions/628/) | 2024 | Vallance, P., Oh, E., Mullins, J., Gulati, M., Hoffman, J., & Carrick, M. (2024, September). TorchSig: A GNU Radio Block and New Spectrogram Tools for Augmenting ML Training. In Proceedings of the GNU Radio Conference (Vol. 9, No. 1). |
| [Large Scale Radio Frequency Wideband Signal Detection & Recognition](https://doi.org/10.48550/arXiv.2211.10335)| 2022 | Boegner, L., Vanhoy, G., Vallance, P., Gulati, M., Feitzinger, D., Comar, B., & Miller, R. D. (2022). Large Scale Radio Frequency Wideband Signal Detection & Recognition. arXiv preprint arXiv:2211.10335. |
| [Large Scale Radio Frequency Signal Classification](https://doi.org/10.48550/arXiv.2207.09918) | 2022 | Boegner, L., Gulati, M., Vanhoy, G., Vallance, P., Comar, B., Kokalj-Filipovic, S., ... & Miller, R. D. (2022). Large Scale Radio Frequency Signal Classification. arXiv preprint arXiv:2207.09918. |


# Citing TorchSig

Please cite TorchSig if you use it for your research or business.

```bibtext
@misc{torchsig,
  title={Large Scale Radio Frequency Signal Classification},
  author={Luke Boegner and Manbir Gulati and Garrett Vanhoy and Phillip Vallance and Bradley Comar and Silvija Kokalj-Filipovic and Craig Lennon and Robert D. Miller},
  year={2022},
  archivePrefix={arXiv},
  eprint={2207.09918},
  primaryClass={cs-LG},
  note={arXiv:2207.09918}
  url={https://arxiv.org/abs/2207.09918}
}
```
