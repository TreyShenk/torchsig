# TorchSig Signal-Processing Review Status

Originally reviewed against `main` at commit `9b1949e` (`v2.1.1`). The focus of this review was signal-generation correctness, physical units, impairment models, bandwidth/SNR metadata, and the consistency of dataset labels with generated IQ data. Five sets of clear corrections, including clean equal-tail occupied-bandwidth estimation and generation-fidelity defaults, are now on `main`.

## Maintenance workflow

- After a feature or correction is complete and validated, update this file, commit the change, and fast-forward the fork's `main` branch so the corrected package is immediately installable.
- Push only to the user's repository. Fetch and push access to the official TorchSig repository remain disabled unless the user explicitly authorizes official-repository interaction.
- Keep unresolved modeling decisions and future technical direction documented here rather than applying speculative partial fixes.
- The README summarizes the fork's independent scope, install URL, user-visible calibration changes, clean-versus-augmentation behavior, and remaining modeling limits. Keep implementation detail and the full decision record here.

## Overall assessment

The signal models are not yet trustworthy as physically calibrated generators. Several default impairment paths contain definite unit or modeling errors, and bandwidth does not have one consistent meaning throughout the codebase. These issues can affect both the IQ data and the labels used to train downstream models.

## Implementation progress

The first set of small, unambiguous P1 corrections is on `main`:

- **Resolved:** Coarse gain now converts dB to sample amplitude with `10 ** (dB / 20)`.
- **Resolved:** ChirpSS now treats `bandwidth` as the total sweep width and chirps from `-bandwidth / 2` to `+bandwidth / 2`.
- **Resolved:** IQ amplitude imbalance now applies symmetric differential half-gains. The requested value is the I-to-Q amplitude ratio in dB, while geometric-mean gain remains one.
- **Resolved:** Digital AGC now uses actual amplitude dB throughout, and its overflow-rate distribution now uses the configured overflow parameter.

The following small, unambiguous P2 corrections are also on `main`:

- **Resolved:** Dataset frequency-overlap rectangles now use the physical edges `center_freq ± bandwidth / 2` and map the `Fs`-wide Nyquist interval to FFT bins with an `Fs` denominator.
- **Resolved:** Integer polyphase decimation now preserves unity DC gain. The obsolete AM factor-of-two compensation and FSK bandwidth-dependent gain multiplier were removed with it.
- **Resolved:** Frequency-edge setters now pass lower and upper edges in the correct order and snapshot the unchanged edge before updating coupled center-frequency and bandwidth metadata.

The clean occupied-bandwidth correction is also on `main`:

- **Resolved:** The generated component's `bandwidth` is now the contiguous FFT-bin interval containing 99% of its clean, time-averaged spectral power, using equal 0.5% frequency tails. The measurement occurs before SNR scaling and before the component is mixed into the dataset noise floor, so it is independent of target SNR and noise level.

The dataset-augmentation separation correction is also on `main`:

- **Resolved:** Impairment level 0 now produces clean, unmodified dataset IQ. Levels 1 and 2 contain physical receiver impairments only; they no longer silently add ML-training augmentations.
- **Resolved:** The former `RandAugment` set is available explicitly as `MLAugmentations`. It contains `RandomDropSamples`, `ChannelSwap`, `TimeReversal`, and `AddSlope`, and is documented as unsuitable for physically calibrated generation or exact-label evaluation unless the caller has accounted for the applied transform.

The following P1 items were deliberately deferred:

- **Deferred — clock jitter/drift:** Multiplying the ppm fraction by the nominal commutator step would fix the obvious scale error, but the intended stochastic models still need to be defined. In particular, jitter should represent sampling-time error rather than an unexplained rate perturbation, and drift needs a documented relationship among oscillator error, accumulated timing phase, output length, and interpolation phase. No partial correction was made.

The following P2 items remain deliberately deferred:

- **Deferred — FM bandwidth:** Correct normalization depends on defining whether the target is peak deviation, Carson bandwidth, 3 dB bandwidth, or occupied-power bandwidth, and on choosing how a stochastic Gaussian message is bounded. This is a waveform-model decision rather than a mechanical correction.
- **Deferred — TX spur and DC-offset reference levels:** The current noiseless generation path has no physical noise reference. A correction requires deciding whether levels are relative to carrier power, an explicit configured noise floor, or a later dataset noise floor.

## High-priority findings

### P1: Clock jitter and drift are scaled about 5,000 times too small

Location: [`torchsig/utils/dsp.py`](torchsig/utils/dsp.py), lines 827–860.

`jitter_std` and `drift_std` are fractional ppm values, but they are added directly to `drate`, whose default value is 5,000. A 10 ppm perturbation should be approximately:

```text
5000 * 10e-6 = 0.05 commutator units
```

The implementation adds `1e-5`, making the requested error 5,000 times too small. Because the polyphase branch is selected with `int(q_step)`, the default perturbations often never select a different branch. Existing tests compare impaired data with the original input; filtering in the resampler can make that comparison pass even if jitter itself has no effect.

### P1: IQ amplitude imbalance does not create an amplitude imbalance — resolved

Location: [`torchsig/transforms/functional.py`](torchsig/transforms/functional.py), lines 751–759.

Status: Resolved on `main` using symmetric differential half-gains.

Both I and Q receive the same multiplier:

```python
10 ** (amplitude_imbalance / 10.0)
```

This is common gain, not relative I/Q imbalance, and therefore does not create the expected image component. The expression also uses a power-dB conversion for sample amplitude. Apply differential I/Q gains using `/20`, commonly as symmetric half-gains, so that the requested imbalance corresponds to a measurable image-rejection ratio.

### P1: ChirpSS occupies twice its declared bandwidth — resolved

Location: [`torchsig/signals/builders/chirpss.py`](torchsig/signals/builders/chirpss.py), lines 73–75.

Status: Resolved on `main` by using `±bandwidth / 2` sweep endpoints.

The chirp runs from `-bandwidth` to `+bandwidth`, giving a total frequency span of `2 * bandwidth`. Resampling and metadata treat `bandwidth` as the total occupied width. A numerical instantaneous-frequency check produced a span of approximately `0.4999 Fs` for a declared bandwidth of `0.25 Fs`.

Use `-bandwidth / 2` through `+bandwidth / 2`, unless the metadata convention is deliberately changed everywhere else.

### P1: The bandwidth estimator is not a 99% occupied-bandwidth estimator — resolved

Location: [`torchsig/utils/dsp.py`](torchsig/utils/dsp.py), lines 1461–1502.

The estimator selects every max-hold bin above `noise_power_db + 3 dB` after calibrating SNR using the maximum of a time-averaged spectrum. Mixing these two spectral statistics makes the reported bandwidth depend directly on SNR and burst-to-burst variation. For the same requested 1 MHz waveforms, live measurements produced:

| Waveform | Bandwidth at 0 dB SNR | Bandwidth at 50 dB SNR |
| --- | ---: | ---: |
| Rectangular QPSK | 1.33 MHz | 6.56 MHz |
| 64-subcarrier OFDM | 1.04 MHz | 4.75 MHz |

At high SNR, far-out sidelobes and leakage cross the absolute threshold. At low SNR, max-hold bins can still cross it even though the time-averaged spectrum used for calibration is lower.

No spectral-power integration is performed, despite the `bandwidth99` name. As a result, bounding-box bandwidth varies with SNR and waveform sidelobes instead of following a consistent occupied-power definition. If the intended quantity is visible-above-noise bandwidth for image labels, it should be named and documented as such rather than replacing physical bandwidth metadata.

Status: Resolved on `main`. The estimator averages the clean spectrogram in linear-power units and retains the contiguous FFT-bin interval after removing 0.5% of power from each frequency tail. The result is computed before SNR scaling and before additive noise.

#### Bandwidth follow-up decisions

The following direction was agreed after implementing the minimal correction:

- Ground-truth bandwidth is the realization-specific 99% equal-tail occupied bandwidth measured from the isolated clean component after component transforms and before SNR scaling or additive dataset noise.
- Ground-truth bandwidth must not depend on SNR, a detector threshold, spectrogram display settings, or any other detection scheme.
- SNR-dependent visible-above-noise bandwidth is explicitly rejected as a ground-truth field and is no longer a proposed metadata direction.
- Variation among stochastic clean realizations is legitimate ground-truth variation rather than a reason to substitute an SNR-dependent measurement.

The minimal correction deliberately retains the existing `bandwidth` field and existing Blackman-window STFT configuration. The following questions remain for later work:

- Decide whether to preserve the generator input separately as `requested_bandwidth` for provenance and validation. It should not be described as universally exact theoretical bandwidth or used as the example-level ground-truth label.
- Determine which waveform families have useful analytical bandwidth predictions. Use those predictions as family-specific sanity checks or ensemble validation, not as a universal replacement for clean realization measurements.
- Decide whether labels should describe intrinsic steady-state modulation bandwidth or the emitted finite burst, whose time gating can broaden the spectrum.
- Quantify estimator sensitivity to FFT size, stride, Blackman windowing, short bursts, and bin-edge quantization.
- Decide whether asymmetric signals need stored lower/upper occupied-frequency edges in addition to width; equal tails are retained for now rather than a minimum-width interval.
- Decide whether bandwidth-changing channel or receiver transforms should produce a second clean received-bandwidth field. The current measurement is after component transforms but before frequency translation, dataset noise, and dataset-level transforms.
- Where a defensible analytical prediction exists, validate stochastic generators by comparing occupied-bandwidth distributions across seeds with that family-specific expectation rather than requiring every realization to match exactly.

### P1: Coarse gain changes double the requested dB change — resolved

Location: [`torchsig/transforms/functional.py`](torchsig/transforms/functional.py), lines 404–409.

Status: Resolved on `main` with amplitude-dB conversion.

IQ samples are amplitudes, so a dB gain must use:

```python
10 ** (gain_change_db / 20)
```

The current `/10` expression makes a 20 dB step multiply amplitude by 100, producing a 40 dB power change. This transform is part of the default receiver impairment path.

### P1: Digital AGC parameters are labeled as dB but implemented as nepers — resolved

Locations:

- [`torchsig/transforms/functional.py`](torchsig/transforms/functional.py), lines 529–548.
- [`torchsig/transforms/transforms.py`](torchsig/transforms/transforms.py), lines 925–930.

Status: Resolved on `main`; the AGC now uses amplitude dB consistently and reads its configured overflow rate.

Signal level and gain use `np.log` and `np.exp`, while the public parameters, thresholds, and documentation call the values dB. Consequently, a nominal 5 dB interval is actually 5 nepers, or about 43.4 dB. `initial_gain_db` is likewise exponentiated directly.

The implementation should either consistently use `20 * log10(amplitude)` and `10 ** (gain_db / 20)`, or rename and retune every value as nepers.

Separately, `alpha_overflow_distribution` is constructed from `self.alpha_track` instead of `self.alpha_overflow`. The intended default overflow response of 0.1–0.3 is therefore replaced by approximately `1e-6`–`1e-5`.

## Medium-priority findings

### P2: FM bandwidth is not controlled by the Carson's-rule calculation

Location: [`torchsig/signals/builders/fm.py`](torchsig/signals/builders/fm.py), lines 63–82.

`fdev` is calculated as a peak frequency deviation, but it multiplies a Gaussian message normalized before low-pass filtering. Filtering changes the source amplitude, and a Gaussian source has no bounded peak. The effective deviation and resulting occupied bandwidth therefore do not match the requested value in a controlled way.

Normalize or bound the filtered source before applying `fdev`, and explicitly define whether the metadata represents Carson bandwidth, 3 dB bandwidth, or an occupied-power bandwidth.

### P2: Frequency-overlap rectangles are twice as wide as the physical signal — resolved

Location: [`torchsig/datasets/datasets.py`](torchsig/datasets/datasets.py), lines 469–478.

A signal with bandwidth `B` should cover `center_freq ± B/2`. The overlap calculation uses `center_freq ± B`, causing placement to reject signals that are actually spectrally disjoint. Its frequency normalization also divides by `Fs/2` rather than `Fs`; in conventional normalized coordinates the rectangle height is four times the expected value, while its physical frequency interval is twice as wide.

Status: Resolved on `main` using `center_freq ± bandwidth / 2` and normalizing the full Nyquist interval by `Fs`.

### P2: Decimation and FSK apply inconsistent amplitude scaling — resolved

Locations:

- [`torchsig/utils/dsp.py`](torchsig/utils/dsp.py), lines 572–588.
- [`torchsig/signals/builders/fsk.py`](torchsig/signals/builders/fsk.py), lines 223–226.

The decimation prototype is divided by the decimation factor, producing DC gain `1/M`; a standard decimation filter should preserve unity DC gain. AM manually compensates for its factor-of-two case, but the general resampler does not.

FSK introduces a second bandwidth-dependent amplitude change with:

```python
fsk_correct_bw *= 1 / resample_rate_ideal
```

The interpolator has already applied its required gain. Later SNR correction can conceal this issue in generated datasets, but it affects standalone modulators and any nonlinear impairment applied before SNR normalization.

Status: Resolved on `main`. The decimation prototype retains unity DC gain, the AM-specific compensation was removed, and FSK no longer applies a resampling-rate-dependent amplitude multiplier.

### P2: TX spur and DC-offset levels are referenced to a nonexistent noise floor

Locations:

- [`torchsig/transforms/functional.py`](torchsig/transforms/functional.py), lines 763–785.
- [`torchsig/transforms/functional.py`](torchsig/transforms/functional.py), lines 1494–1524.

The TX impairment path applies these operations to isolated, noiseless generated signals without supplying `noise_power_db`. The functions estimate a noise floor from the minimum of the signal spectrum. Deterministic spectra can contain exact or near-exact FFT zeros, producing `-inf` or arbitrary numerical-floor estimates and causing nominal impairments to disappear or vary unpredictably.

### P2: Frequency-edge setters can create negative bandwidths — resolved

Location: [`torchsig/signals/signal_types.py`](torchsig/signals/signal_types.py), lines 138–188.

The `upper_freq` and `lower_freq` setters pass upper and lower edges to `bandwidth_from_lower_upper_freq` in reverse order. If these setters are used, the resulting bandwidth can be negative.

Status: Resolved on `main`. The setters also retain the cached opposite edge before updating bandwidth so the subsequent center-frequency calculation cannot observe partially updated metadata.

## External validation findings and generation-fidelity status

Source: PyRISE detector/localizer validation notes produced while using this fork as a clean labeled signal source at commit `625f852`.

### Resolved — constellation pulse-shape default

`ConstellationSignalGenerator` selects rectangular and SRRC symbol pulses with equal probability. The rectangular pulse is a valid baseband model, but without a subsequent transmit filter it produces broad sinc sidelobes and is a poor 50% default for an off-air communications dataset. The clean occupied-bandwidth labels correctly expose this generated energy; the label is not the defect.

Status: Resolved on `main`. Dataset-generated PSK/QAM/ASK signals now always use SRRC shaping with an independently drawn rolloff in the existing 0.1–0.5 range. Explicit `"rectangular"` support remains in `constellation_modulator_baseband()` and `constellation_modulator()` for intentional experiments. Regression tests prove both behaviors.

Future validation: compare occupied-bandwidth distributions across constellation families and seeds. That is useful calibration work, but it does not block this focused default correction.

A configurable rectangular-pulse probability may be added later if a concrete dataset need appears; it is not required for the minimal correction.

### Resolved — signal-family alias collisions

Family aliases are currently built with substring matching. This silently overwrites the standalone analog-FM key and creates incorrect memberships:

- `fm` emits `fm`, `lfm-data`, and `lfm-radar`.
- `am` emits the analog-AM variants plus every QAM class because `qam` contains the substring `am`.

Status: Resolved on `main`. Public family aliases now use explicit memberships rather than substring matching: `fm` remains standalone analog FM, `lfm` contains only `lfm-data` and `lfm-radar`, and `am` contains only the four analog-AM modes. `ofdm`, `fsk`, `psk`, `qam`, `ask`, and `msk` also have explicit, regression-tested membership lists.

No combined analog-FM/LFM family is planned; those waveforms are operationally distinct despite both using frequency modulation.

### Resolved — nonphysical ML augmentations were included in level-0 data

Source: PyRISE detector/localizer validation notes produced at commit `b119f7f`.

The former level-0 `Impairments.dataset_transforms` selected two of `RandomDropSamples`, `ChannelSwap`, `TimeReversal`, and `AddSlope` for every sample. Although these can be useful training augmentations, they are not a physical receiver or propagation model. In particular, `AddSlope` implements `y[n] = 2x[n] - x[n-1]`, whose power response is `5 - 4 cos(2πf)`: its relative minimum is at DC and it rises by about 9.5 dB toward the band edges. It therefore caused a randomly occurring broad DC bowl in the noise floor.

Status: Resolved on `main`. Physical impairment levels are now augmentation-free, so level 0 matches its documented clean-IQ contract. The augmentation set is available only through an explicit `MLAugmentations()` transform supplied by the caller. This separates robustness-training data from calibrated signal-generation and ground-truth evaluation data.

The expected probability of the former DC-bowl effect was 50%, because `RandAugment` selected two transforms uniformly without replacement from four candidates. A measured rate near that value across a finite validation corpus is expected.

Follow-up policy:

- Use `default_dataset(impairment_level=None)` or `default_dataset(impairment_level=0)` for clean generated IQ. `TorchSigDataModule` defaults to the latter and is now also clean.
- Add `MLAugmentations()` explicitly only for training workflows that intentionally value augmentation over physical fidelity.
- Do not treat labels as exactly calibrated after an augmentation that changes a signal's spectral distribution without a corresponding metadata update. `AddSlope` is the primary example.
- `PassbandRipple` remains a separate physical-model question: at levels 1 and 2 it can intentionally color the complete capture by roughly 1–2 dB. Its response and gain/SNR convention require a later design decision rather than a partial removal.

### Decision required — OFDM cyclic-prefix model

OFDM currently omits the cyclic prefix approximately 50% of the time. The variable name is misleading: the code sets `cp_len = 0` when the probability condition succeeds, so simply raising `cyclic_prefix_probability` to 1.0 would make every signal CP-less. The current nonzero CP length is also drawn uniformly from 2 through nearly half the subcarrier count, which is not a clearly realistic deployed-system distribution.

Before implementation, decide:

1. Whether the generic `ofdm-*` classes should model deployed CP-OFDM by default while retaining an explicit CP-less option.
2. Whether CP lengths should be selected from conventional fractions such as 1/4, 1/8, 1/16, and 1/32 of the useful symbol duration.
3. Whether CP presence and allowed ratios should be generator metadata/configuration rather than hidden constants.

The likely direction is CP-bearing OFDM by default with explicit, correctly named configuration, but no partial probability-only fix should be made before the length model is chosen.

### Reviewed — plain FSK/MSK pulse shape is not a defect

Plain FSK and MSK use a rectangular instantaneous-frequency pulse with continuous phase accumulation. This is a standard full-response CPFSK model; MSK specifically uses the rectangular frequency pulse with modulation index 0.5. Gaussian shaping intentionally defines the separate GFSK/GMSK variants.

No signal-generation change is planned. The distinction may be documented more explicitly, and additional shaped-FSK variants should use explicit names rather than silently changing plain FSK/MSK semantics.

### Scenario configuration and optional future features

- `cochannel_overlap_probability` is an overlap-acceptance probability per candidate placement, not the expected fraction of completed samples containing overlap. A first-class minimum time/frequency separation or guard-band option could improve controlled detector validation, but setting overlap probability to zero is sufficient for the current harness.
- Signal palette, SNR distribution, and bandwidth range are scenario choices. The consuming project should continue to configure them rather than this fork imposing one communications-only global default.
- Named curated palettes and a minimum-separation option remain possible convenience features, not correctness fixes.

## Validation status

- Repository syntax compilation passed with `python3 -m compileall`.
- An isolated environment was installed with `uv` using CPython 3.14.4.
- The original targeted transform suite passed before implementation: 7 tests passed and 80 were deselected.
- After implementation, 11 targeted functional/regression tests and 13 wrapper/impairment tests passed.
- The full suite after implementation completed with 265 tests passed, 1 failed, and 3 deselected. The failure is in the two-worker dataset test because it passes a local lambda to a spawned worker and the lambda cannot be pickled; it is unrelated to these DSP changes.
- After the second correction pass, the full suite completed with 270 tests passed, the same 1 unrelated multiprocessing/lambda failure, and 3 deselected.
- The generation-fidelity correction tests passed: 11 targeted tests and 175 combined signal/transform tests passed.
- A full-suite run after the generation-fidelity correction completed with 265 tests passed and 3 deselected. The remaining 19 failures are all multiprocessing dataset-loader cases: this macOS sandbox denies PyTorch's `torch_shm_manager` shared-memory helper (`Operation not permitted`) when tests spawn workers. They do not exercise the changed signal-generation or alias code. Re-run those worker tests on the intended Linux development machine outside this sandbox.
- The augmentation-separation tests verify that level 0 preserves a flat noise spectrum and that the former augmentation set is explicit and reproducible when seeded.
- The full suite after augmentation separation completed with 267 tests passed and 3 deselected. Its same 19 failures are macOS sandbox multiprocessing cases blocked from launching PyTorch's shared-memory helper; the focused physical-impairment, signal/transform, dataset, and writer tests passed.
- Independent physics-based checks produced the following results:

| Check | Requested or expected | Measured |
| --- | ---: | ---: |
| Clock jitter versus zero-jitter baseline | 50 ppm should alter resampling phase | Bit-for-bit identical; 0 changed samples |
| Clock drift versus zero-drift baseline | 10 ppm | Maximum sample difference `3.14e-4` |
| Coarse gain | 20 dB = 10× amplitude | 100× amplitude |
| IQ amplitude imbalance | 1 dB should create a measurable image | Image approximately `-313 dBc`, i.e. numerical noise only |
| ChirpSS/chirp span | Declared `0.25 Fs` | `0.49988 Fs` |
| Factor-of-two decimator DC gain | 1.0 | 0.5 |

- Direct 99%-power FFT measurements for a requested 1 MHz bandwidth gave approximately 0.27–0.33 MHz for FM, 1.20 MHz for SRRC QPSK, 1.96–1.98 MHz for ChirpSS, and 1.01 MHz for 64-subcarrier OFDM. These measurements confirm that the current `bandwidth` field does not represent one common spectral quantity across generators.
- Post-fix measurements confirmed 10× amplitude for a 20 dB coarse gain, exactly 6 dB I-to-Q gain ratio for a requested 6 dB imbalance, 10× AGC amplitude for a 20 dB initial gain, and approximately 0.993 MHz 99%-power bandwidth for a requested 1 MHz ChirpSS waveform.
- The second correction pass has 10 focused DSP regression tests passing and 162 transform/signal tests passing. A factor-of-two decimator now measures unity gain away from filter transients, and FSK RMS amplitude remains within approximately 0.3% of unity at interpolation, pass-through, and decimation bandwidths.
- The equal-tail bandwidth pass has 12 focused DSP regression tests and 164 transform/signal tests passing. The full suite completed with 272 passed, the same 1 unrelated multiprocessing/lambda failure, and 3 deselected. For clean requested-1 MHz examples, the estimator measured approximately 1.230 MHz for SRRC QPSK and 1.035 MHz for both OFDM-64 and ChirpSS; changing target SNR does not change the measured bandwidth.
- Existing transform tests mainly verify shapes, dtypes, and that output differs from input. They generally do not verify occupied bandwidth, image-rejection ratio, calibrated gain, or impairment magnitude against physical expectations.

## Recommended next steps

1. Choose the OFDM cyclic-prefix presence and length model, then implement it as one coherent change. First add a deterministic diagnostic that verifies the DC-subcarrier null remains at the shifted center frequency after resampling.
2. Define and fix the clock jitter/drift model before treating those impairments as physically calibrated.
3. Decide whether to retain `requested_bandwidth` as provenance alongside clean occupied-bandwidth ground truth, and standardize estimator parameters.
4. Continue numerical DSP validation across seeds and waveform families, especially FM calibration.
5. Define the intended receiver-filter model for `PassbandRipple`, including its gain normalization and SNR reference, before considering it calibrated.
6. Revisit existing generated datasets after fixes; corrected waveforms, impairments, and labels will not be statistically compatible with datasets generated by the current code.
