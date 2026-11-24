# Data Card — NASA C-MAPSS FD001

## 1. Dataset summary

| Field | Value |
|---|---|
| Dataset name | Turbofan Engine Degradation Simulation Data Set (C-MAPSS) |
| Subset used | **FD001** |
| Domain | Aircraft turbofan engine prognostics |
| Task | Remaining useful life (RUL) regression |
| Data type | Multivariate time series, simulation output |
| Source | NASA Prognostics Center of Excellence (PCoE) Data Set Repository |
| Original publication | Saxena & Goebel (2008) |
| Accessed | November 2026 |
| Unit of analysis | One engine (unit), observed over successive flight cycles |
| Label | RUL = number of cycles from the current cycle until failure |
| Primary target used here | RUL capped at **125 cycles** |
| Raw files | `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` |

## 2. Motivation

C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) is a NASA
simulator for turbofan engine degradation. The dataset was released to support
prognostics research: given a history of sensor readings, estimate how many
operating cycles remain before the engine must be maintained. It is widely used
as a benchmark for RUL regression, making it suitable for comparing feature
engineering and regression algorithms under a common protocol.

This project uses C-MAPSS because it is public, well documented, multivariate,
and time-dependent. It is also a **simulated** benchmark, so it supports
methodological claims rather than claims about a real fleet.

## 3. Composition

### 3.1 Files used

| File | Contents | Rows |
|---|---|---:|
| `train_FD001.txt` | 100 run-to-failure engine trajectories | 20,631 |
| `test_FD001.txt` | 100 truncated engine trajectories | 13,096 |
| `RUL_FD001.txt` | One ground-truth RUL value per test engine | 100 |

### 3.2 Structure

Each row contains 26 space-separated values:

| Position | Column | Meaning |
|---:|---|---|
| 1 | `unit` | Engine/unit identifier |
| 2 | `cycle` | Operating cycle index within that engine |
| 3–5 | `op1`, `op2`, `op3` | Operational settings |
| 6–26 | `s1`–`s21` | Sensor measurements |

### 3.3 FD001 characteristics

| Property | Value |
|---|---|
| Operating conditions | 1 (sea level) |
| Fault mode | 1 (high-pressure compressor degradation) |
| Train engines | 100 |
| Test engines | 100 |
| Train rows | 20,631 |
| Test rows | 13,096 |
| Train cycles per engine | 128–362 (mean 206.3, median 199) |
| Test cycles per engine | 31–303 (mean 131.0, median 133.5) |
| Raw training RUL range | 0–361 |
| Official test RUL range | 7–145 (mean 75.52, median 86, std 41.76) |

### 3.4 Sensors

All 21 sensors are present, but six are constant in FD001 and carry no
information:

` s1, s5, s10, s16, s18, s19`

The remaining 15 sensors are used:

` s2, s3, s4, s6, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`

Constant sensors are detected from the **100 training engines only** and removed
from both train and test frames. This avoids using test statistics to make
preprocessing decisions. The six sensors have zero variance across all training
engines, so the set is invariant to the 80/20 engine split.

## 4. Collection process

The data were generated with the C-MAPSS simulator, not collected from a
physical fleet. Each training engine starts in a healthy state and is simulated
until failure, producing a complete run-to-failure trajectory. Test engines are
simulated from the same process but truncated before failure; the withheld RUL
at the final observed cycle is supplied in `RUL_FD001.txt`.

The simulator models degradation propagation through the engine's components.
FD001 uses one operating condition and one fault mode, so it is the simplest of
the four C-MAPSS subsets. FD002 and FD004 add six operating conditions; FD003
adds a second fault mode.

## 5. Preprocessing and labeling

### 5.1 Target construction

For a training engine whose final cycle is `T`, the RUL at cycle `t` is:

```text
RUL(t) = T - t
```

The target is capped at 125 cycles:

```text
y(t) = min(T - t, 125)
```

The cap reflects a piecewise-linear health assumption: once an engine is more
than 125 cycles from failure, the exact remaining life is not identifiable from
the available degradation signal, and forcing the model to distinguish 200 from
300 remaining cycles adds noise. It also matches common C-MAPSS practice. The
raw official test RUL is still reported as a secondary metric, so the effect of
the cap is visible.

### 5.2 Splits

- **Training/validation:** 80 training engines for fitting, 20 held-out engines
  for model selection.
- **Validation cutoffs:** each validation engine is truncated at five random
  cycles between 25% and 100% of its lifetime, producing 100 validation
  predictions that resemble the official test protocol.
- **Test:** the 100 official test engines are used only after model selection.
  The model predicts once at each test engine's last observed cycle.
- **Final refit:** the selected model is refit on all 100 training engines
  (including the 20 validation engines) before the official test evaluation.

No row-wise random split is used. Rows from the same engine are highly
correlated; splitting them across train and validation would leak engine-specific
information and inflate scores.

### 5.3 Feature engineering

For each of the 15 retained sensors, the pipeline adds:

- first difference (`sensor_diff`);
- rolling mean, standard deviation, minimum, and maximum over windows of
  10, 30, and 50 cycles.

All rolling statistics are backward-looking (`min_periods=1`), so no future
cycle is used. The three operational settings are kept as raw features. The
optional raw cycle-index feature is disabled by default because it can act as a
shortcut on run-to-failure data. Per-engine normalization is deliberately avoided
because it would use future information.

Result: 213 features (18 raw + 15 differences + 180 rolling statistics).

## 6. Uses

### Intended uses

- Benchmarking RUL regression algorithms under a public protocol.
- Teaching time-series feature engineering, engine-level validation, and
  asymmetric error analysis.
- Demonstrating a decision-support prototype for maintenance prioritization.

### Out-of-scope uses

- Certifying airworthiness or making real maintenance decisions.
- Claiming performance on a physical fleet without external validation.
- Inferring failure modes or diagnosing specific components. The dataset
  provides a single RUL label, not component-level fault labels.
- Redistributing the raw data without checking NASA's terms.

## 7. Limitations and biases

| Limitation | Consequence |
|---|---|
| Simulated data | Sensor noise, degradation physics, and operating profiles are approximations of reality. |
| One operating condition (FD001) | The model may not generalize to changing altitude, temperature, or load. |
| One fault mode | The model learns one degradation pattern, not general engine failure. |
| No maintenance history | Maintenance actions that could reset or alter degradation are absent. |
| No uncertainty labels | Ground truth is a single RUL value; label noise and ambiguity are not quantified. |
| Truncated test labels | The official test set covers only one prediction point per engine. |
| Capped target | The model is intentionally less precise above 125 cycles. |
| Benchmark optimism | Published C-MAPSS scores are not evidence of field performance. |
| Selection on 20 engines | Validation is noisy; the selected model may not be optimal on a different engine split. |

## 8. Ethical considerations

- **Safety:** RUL under-estimation can cause unnecessary maintenance;
  over-estimation can cause in-service failure. The NASA asymmetric score and
  late-prediction rate are reported for this reason. A real system needs
  conservative buffers, human review, and redundancy.
- **Accountability:** The model is decision-support only. A qualified engineer
  remains responsible for maintenance decisions.
- **Transparency:** The pipeline, features, splits, metrics, and limitations are
  documented. The saved model bundle contains its feature list and metadata.
- **Privacy:** The dataset contains no personal data.
- **Research integrity:** C-MAPSS is a public benchmark; results should be
  presented as benchmark results, not as a deployed system.

## 9. Distribution and license

The dataset is distributed by the NASA Prognostics Center of Excellence:

- Landing page:
  `https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data`
- Download mirror used by `scripts/download_data.py`:
  `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`

The NASA repository does not state a standard open-source license on the
dataset page. Treat the data as research data: cite the source, do not present
it as your own, and check the repository terms before redistributing the raw
files. The project's download script fetches the data from the public source
rather than bundling it.

## 10. Citation

> A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data
> Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett
> Field, CA.

Related paper:

> A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for
> Aircraft Engine Run-to-Failure Simulation," International Conference on
> Prognostics and Health Management (PHM), 2008.

## 11. File integrity

SHA-256 checksums of the FD001 files used in this project:

| File | SHA-256 |
|---|---|
| `train_FD001.txt` | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `test_FD001.txt` | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `RUL_FD001.txt` | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |

Verify with:

```bash
sha256sum data/raw/CMAPSS/train_FD001.txt \
          data/raw/CMAPSS/test_FD001.txt \
          data/raw/CMAPSS/RUL_FD001.txt
```

## 12. Maintenance

- Re-download: `python scripts/download_data.py --force`
- The download script checks that all 12 train/test/RUL files (FD001–FD004) are
  present.
- If NASA changes the archive layout, update `extract_outer_and_inner()` in
  `scripts/download_data.py`.
- Re-run `python -m src.train --subset FD001` after any data or feature change.
