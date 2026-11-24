# Viva Preparation — C-MAPSS RUL Project

## 1. One-minute project pitch

> I built a remaining useful life estimation system on the NASA C-MAPSS FD001
> turbofan dataset. The goal is to predict how many cycles an engine can run
> before maintenance. I capped the target at 125 cycles, dropped six constant
> sensors, and engineered 213 features using first differences and rolling
> means, standard deviations, minima, and maxima over 10-, 30-, and 50-cycle
> windows. I split by engine, not by row, and truncated each validation engine
> at five random cycles to mimic the official test set. I compared a mean
> baseline, Ridge, Random Forest, HistGradientBoosting, and XGBoost. Ridge won
> on validation RMSE and NASA score. On the official test set it achieved RMSE
> 14.89 cycles, MAE 11.63, R² 0.862, and NASA score 3.20. An ablation showed
> that the rolling features cut Ridge's test RMSE by about 28% and its NASA
> score by about 74% compared with raw sensors. I also built a Streamlit
> prototype for single-engine, batch, and CSV-upload predictions.

---

## 2. Numbers to memorize

| Item | Value |
|---|---|
| Dataset | NASA C-MAPSS FD001 |
| Train / test engines | 100 / 100 |
| Train / test rows | 20,631 / 13,096 |
| Raw sensors | 21 |
| Constant sensors dropped | 6 (`s1, s5, s10, s16, s18, s19`) |
| Retained sensors | 15 |
| RUL cap | 125 cycles |
| Features | 213 |
| Rolling windows | 10, 30, 50 |
| Split | 80 train / 20 validation engines |
| Validation predictions | 20 engines × 5 cutoffs = 100 |
| Selected model | Ridge (α = 1.0) |
| Validation RMSE | 14.458 |
| Test RMSE (capped) | 14.892 |
| Test MAE (capped) | 11.635 |
| Test R² (capped) | 0.862 |
| Test NASA mean (capped) | 3.200 |
| Test late predictions | 50 / 100 |
| Test RMSE (raw RUL) | 16.301 |
| Raw-sensor Ridge test RMSE | 20.586 |
| Ablation RMSE reduction | 27.7% |
| Ablation NASA reduction | 74.0% |

---

## 3. Core questions

### Q1. What is remaining useful life?

RUL is the number of operating cycles (or time) left before an asset fails or
must be maintained. For an engine that fails at cycle `T`, RUL at cycle `t` is
`T - t`. It is a regression target, not a class.

### Q2. Why did you choose C-MAPSS?

It is a public, well-documented, multivariate time-series benchmark with
run-to-failure trajectories and official test labels. It is small enough to run
on a laptop and difficult enough to require feature engineering and careful
validation. It is also widely used, so results can be compared with the
literature.

### Q3. Why FD001 and not FD002 or FD004?

FD001 has one operating condition and one fault mode, so it isolates the
feature-engineering and validation methodology. FD002/FD004 have six operating
conditions, where raw sensor values shift with regime; they need condition-aware
normalization. FD003 has two fault modes. I documented them as extensions rather
than pretending FD001 covers them.

### Q4. Why cap RUL at 125?

Early in an engine's life, the sensors are nearly flat and the exact RUL (say,
250 vs 300) is not identifiable. Capping creates a piecewise-linear target and
focuses the model on the final 125 cycles, where degradation is visible. This is
standard in C-MAPSS work. The trade-off is that predictions are clipped at 125,
so I also report raw-target metrics.

### Q5. Why not randomly split rows into train and test?

Rows from the same engine are highly correlated. A random row split would put
near-identical cycles from the same engine in both sets, leaking engine-specific
information and inflating the score. The correct split is by engine unit.

### Q6. How did you build the validation set?

I held out 20 of the 100 training engines. For each held-out engine, I truncated
its trajectory at five random cycles between 25% and 100% of its life. That
gives 100 validation predictions with a mix of RUL values. Evaluating at the
last cycle of a full run-to-failure engine would always give RUL = 0, which is
not representative.

### Q7. Why five truncation points per engine?

One cutoff per engine gives only 20 validation predictions and is noisy. Five
cutoffs give 100 predictions and reduce selection variance, while still keeping
all rows from an engine on one side of the split.

### Q8. What features did you engineer?

For each of 15 sensors: first difference, rolling mean, standard deviation,
minimum, and maximum over windows of 10, 30, and 50 cycles. Plus the three raw
operational settings. That is 213 features.

Calculation:

```text
18 raw + 15 diffs + 15 × 4 statistics × 3 windows = 213
```

### Q9. Why those window sizes?

10 cycles captures short-term noise and recent change; 30 cycles captures
medium-term degradation; 50 cycles captures the longer trend. Using multiple
scales lets the model see both local and global behaviour. The windows are
hyperparameters, but the validation results were stable and I did not tune them
extensively.

### Q10. Do the rolling features use future data?

No. Every rolling window ends at the current cycle (`min_periods=1`), and the
first difference is backward-looking. A unit test checks that the first row's
rolling mean equals its own sensor value, not a future one.

### Q11. Why is `cycle` excluded as a feature?

On run-to-failure data, cycle index is a proxy for the label. If the model sees
cycle 200, it can infer a lot about RUL without learning degradation. That can
inflate validation and fail on truncated test engines. I provide an optional
`--include-cycle` flag but keep it off by default.

### Q12. Why did Ridge beat Random Forest and XGBoost?

FD001 has one condition and one fault mode. The rolling means produce a smooth,
nearly monotonic degradation signal, and the relationship with capped RUL is
close to linear. With only 80 training engines and many correlated features, a
regularized linear model has lower variance. The tree models can fit
nonlinearities but overfit engine-specific quirks.

### Q13. Is that not suspicious? Usually gradient boosting wins.

Not necessarily. Feature engineering changed the problem: the rolling means
already encode the nonlinear trend, so the remaining mapping is close to linear.
The ablation shows the features help both model families. A simpler model that
wins is a valid result, and it is easier to explain and deploy.

### Q14. Why include a mean baseline?

To prove the models learn something. The mean baseline has RMSE 42.05 and NASA
score 425.9 on validation. Every real model beats it by a large margin. Without a
baseline, a score of 14.9 has no context.

### Q15. What is the NASA/PHM score?

An asymmetric error score. With `d = prediction - truth`:

```text
exp(-d/13) - 1   if d < 0   (early)
exp( d/10) - 1   if d >= 0  (late)
```

Lower is better. Late predictions are penalized more because overestimating RUL
can lead to in-service failure. Early predictions cause unnecessary maintenance,
which is costly but safer.

### Q16. Why not just use RMSE?

RMSE treats early and late errors symmetrically. In maintenance, the two
directions have different consequences. The NASA score captures that asymmetry,
and the late-prediction rate gives an intuitive count. Reporting RMSE, MAE, R²,
NASA score, and late rate together gives a fuller picture.

### Q17. What does R² = 0.862 mean?

The model explains about 86.2% of the variance in capped test RUL. It is high,
but R² alone can hide large errors in specific ranges, which is why I also report
RMSE, MAE, bins, and worst cases.

### Q18. Why is test RMSE (14.89) close to validation RMSE (14.46)?

That is a good sign. It suggests model selection did not overfit the validation
engines and the test distribution is similar. A large gap would be a warning.

### Q19. The late rate is 50%. Is the model bad?

No. A 50% late rate with no bias adjustment is expected for a roughly centered
predictor; the mean error is −1.32 cycles, so it is not systematically optimistic.
The important point is that late errors are asymmetric in cost. A deployment
would choose a conservative bias or decision rule using real cost data. The
report shows the trade-off; it does not claim 50% is optimal.

### Q20. Why are raw-target metrics worse?

Eleven test engines have true RUL above 125, but the model is capped at 125.
Those predictions are necessarily under-estimates, so raw RMSE rises from 14.89
to 16.30. This is a consequence of the target definition, not a separate bug.
Reporting both makes the trade-off transparent.

### Q21. What did the ablation show?

With raw sensors only, Ridge test RMSE is 20.59 and NASA score is 12.29. With
engineered features, Ridge test RMSE is 14.89 and NASA score is 3.20. That is a
27.7% RMSE reduction and a 74.0% NASA reduction. HistGradientBoosting also
improves, so the gain is from the features, not one algorithm.

### Q22. What are the most important features?

Permutation importance ranks `s12_mean_50`, `s4_mean_30`, `s12_mean_30`,
`s11_mean_30`, and `s11_mean_50` at the top. They are all rolling means, which
matches the ablation: smoothed degradation signals matter more than raw
instantaneous values. Correlated sensors share importance, so the ranking is not
causal.

### Q23. What are the biggest errors?

Unit 12: true 124, predicted 81.8 (42.2 cycles early). Unit 21: true 57,
predicted 94.6 (37.6 cycles late). Most large errors are under-predictions on
high-RUL engines because of the cap and regression to the mean. The dangerous
ones are mid-life late predictions such as units 21 and 67.

### Q24. How did you prevent leakage?

- Split by engine, not row.
- Test labels used only after model selection.
- Constant sensors and imputers fitted on training data only.
- Rolling features past-only.
- Unit ID excluded.
- Cycle excluded by default.
- Preprocessing inside pipelines.
- Validation engines truncated before evaluation.
- Unit tests for target monotonicity and feature alignment.

### Q25. Why did you refit on all 100 training engines?

After selecting the model on 80 training + 20 validation engines, the validation
labels become legitimate training data. Refitting on all 100 uses more data and
is standard. The official test set remains untouched.

### Q26. How is the model saved?

As a joblib bundle containing the fitted pipeline, feature list, retained
sensors, window sizes, RUL cap, subset, selected model name, metrics, and
training timestamp. Inference loads the bundle and rebuilds features from the
stored metadata, so feature names are not hard-coded in the app.

### Q27. How does the prototype work?

Streamlit app with four views:

1. **Engine demo:** select a test engine, get RUL, priority, actual RUL, late
   warning, planning window, and sensor trends.
2. **Batch predictions:** score all test engines, count priorities, download CSV.
3. **Upload:** score a user CSV with the required columns.
4. **Model & metrics:** show test metrics, validation comparison, and figures.

Priority: High ≤ 20 cycles, Medium ≤ 50, Low > 50. Planning window subtracts a
10% buffer (minimum 2 cycles).

### Q28. Why a 10% buffer?

It is a simple, transparent safety margin, not a tuned policy. It pushes the
planning window earlier, which is safer under asymmetric costs. A real system
should tune the buffer using failure cost, downtime cost, and inspection
capacity.

### Q29. What are the main limitations?

Simulated data; one operating condition; one fault mode; no uncertainty
intervals; capped target; small validation set; post-hoc test ablation; fixed
priority thresholds; no drift monitoring; classical models only. The results are
benchmark results, not field guarantees.

### Q30. What are the ethical concerns?

RUL predictions can influence aircraft maintenance. A late prediction can
contribute to in-service failure. The system must be decision-support with human
oversight, conservative buffers, and uncertainty reporting. It must not be
presented as airworthiness-certified. The dataset has no personal data, but the
raw data should be cited and not redistributed without checking NASA's terms.

### Q31. How would you handle FD002/FD004?

Those subsets have six operating conditions. Raw sensor values shift with
altitude, temperature, and load. Options:

- condition-wise normalization (subtract the mean per operating regime);
- cluster operating settings and train regime-specific models;
- add regime indicators or interaction features;
- use a model that is robust to regime shift.

FD001 results do not transfer directly.

### Q32. Why not use an LSTM?

The project goal was a transparent, reproducible baseline with strong feature
engineering. An LSTM would be a natural extension, but it needs more tuning,
more compute, and careful sequence handling. The ablation shows the feature-based
approach is strong, so it is a useful benchmark for any deep model. I list LSTM,
TCN, and Transformer as future work.

### Q33. How would you quantify uncertainty?

Options:

- quantile regression (predict 10th/50th/90th percentiles);
- conformal prediction with a calibration set;
- ensemble variance across seeds/models;
- Bayesian regression or Monte Carlo dropout.

The app currently shows a point prediction plus a fixed buffer, not a calibrated
interval. That is a limitation.

### Q34. How would you set maintenance thresholds?

Use a cost model:

```text
expected cost = P(failure before maintenance) × failure_cost
              + P(unnecessary maintenance) × maintenance_cost
```

Choose the threshold that minimizes expected cost, subject to safety
constraints. The 20/50-cycle thresholds in the demo are illustrative.

### Q35. What happens if a sensor is missing at inference?

The pipeline uses median imputation fitted on training data, so a missing value
can be filled. But a persistently missing or stuck sensor is a data-quality
problem; a production system should flag it rather than silently impute. The
current app validates required columns but does not monitor sensor health.

### Q36. Why median imputation and not mean?

Median is more robust to outliers. FD001 has no missing values, so this is a
robustness measure for other subsets or user uploads.

### Q37. Why scale features for Ridge but not for trees?

Ridge uses a linear combination with a shared penalty, so features on different
scales would be penalized unequally. Standardization puts them on comparable
scales. Trees split on thresholds and are invariant to monotonic scaling, so
scaling is unnecessary.

### Q38. What does Ridge alpha do?

Alpha controls L2 regularization strength. Larger alpha shrinks coefficients
more, reducing variance but increasing bias. I swept 0.01 to 1000; the validation
curve was flat between 0.01 and 3, so I kept α = 1.0.

### Q39. Why both HistGradientBoosting and XGBoost?

They are different implementations of gradient boosting. scikit-learn's
HistGradientBoosting is a fast histogram-based baseline; XGBoost is a widely used
industrial library with different regularization and splitting. Including both
shows the comparison is not tied to one library and satisfies the multi-algorithm
requirement.

### Q40. How do you know the model is not memorizing engine identity?

`unit` is not a feature. The split is by engine. Rolling features are computed
within each engine, so there is no cross-engine leakage. The model sees only
sensor history and settings at prediction time.

### Q41. What tests did you write?

Seven unit tests:

1. RUL target is capped, ends at zero, and is non-increasing.
2. Constant sensors are dropped from both train and test.
3. Engineered features align between train and test.
4. Rolling features do not use future values.
5. The optional cycle feature is the raw index, not a future-normalized value.
6. NASA score penalizes late predictions more than early ones.
7. Prediction-table construction marks late predictions correctly.

### Q42. What would invalidate your results?

- A row-wise split (leakage).
- Using test RUL during training or selection.
- Computing rolling features with future values.
- Dropping sensors based on test statistics.
- Evaluating only at RUL = 0.
- Reporting only the best metric.
- Claiming field performance from simulated data.

### Q43. What is the business value?

Better maintenance timing can reduce unplanned downtime, avoid unnecessary
replacement, and improve safety. The model gives a prioritized list of engines
and a planning window. The value depends on the cost of failure versus the cost
of early maintenance, which is why a cost-sensitive policy is future work.

### Q44. What is the most important figure?

The predicted-vs-true plot and the error-by-RUL-range plot. The first shows
overall fit and the high-RUL compression; the second shows where the model is
accurate and where late predictions cluster. The ablation figure is the most
important for justifying the feature engineering.

### Q45. If you had one more month, what would you do?

1. Extend to FD002/FD004 with operating-condition normalization.
2. Add sequence models and compare fairly.
3. Add calibrated prediction intervals.
4. Build a cost-sensitive maintenance policy.
5. Use repeated engine-level splits for robust selection.
6. Add drift monitoring and a retraining pipeline.
7. Validate on real fleet data if available.

---

## 4. Likely examiner traps

| Trap | Good response |
|---|---|
| "Your test RMSE is higher than validation; is that overfitting?" | The gap is small (14.89 vs 14.46). The raw-target RMSE is higher because of the 125 cap and 11 engines above 125. |
| "Why is Ridge better than XGBoost? Did you tune XGBoost?" | The features already encode the trend; with 80 engines, the linear model has lower variance. I did a small alpha sweep for Ridge, not an extensive tree search, to avoid overfitting the validation set. The ablation shows the feature effect is consistent across model families. |
| "Your late rate is 50%; isn't that dangerous?" | The mean error is slightly early (−1.32), so the model is not systematically late. A 50% rate with no bias adjustment is expected for a centered model. The cost asymmetry is why I report NASA score and late rate, and why the app uses a safety buffer. |
| "Why not use the last cycle of every training engine for validation?" | That would give RUL = 0 for every engine and would not resemble the truncated test set. I truncate at random cycles. |
| "Did you use the test set for feature selection?" | No. Features and model selection use only training/validation engines. The ablation is a post-hoc analysis and is labeled as such. |
| "Why cap at 125 and then complain about raw-target error?" | The cap is a modeling choice. I report both metrics so the cost is visible. The primary result is the capped target, which matches training. |
| "Is C-MAPSS real data?" | No, it is simulator output. The report states this clearly and does not claim field performance. |
| "What is the difference between this and fault classification?" | The target is cycles-to-failure, a continuous regression label. There is no fault-class label per test engine. The evaluation uses regression metrics and asymmetric RUL cost. |

---

## 5. Whiteboard derivations

### RUL target

```text
T_i = max cycle for engine i
RUL(i,t) = T_i - t
y(i,t) = min(RUL(i,t), 125)
```

### RMSE

```text
RMSE = sqrt((1/n) Σ (y_i - ŷ_i)²)
```

### MAE

```text
MAE = (1/n) Σ |y_i - ŷ_i|
```

### NASA score

```text
d_i = ŷ_i - y_i
s_i = exp(-d_i/13) - 1   if d_i < 0
s_i = exp( d_i/10) - 1   if d_i >= 0
Score = (1/n) Σ s_i
```

### Ridge objective

```text
min_w  Σ (y_i - wᵀx_i)² + α ||w||²
```

### Feature count

```text
18 raw + 15 diffs + 15 × 4 × 3 = 213
```

---

## 6. Demo script (3 minutes)

1. **Open the README** and state the problem and headline result.
2. **Show the pipeline diagram** and explain the engine-level split.
3. **Show the model comparison figure** and explain why Ridge won.
4. **Show the predicted-vs-true plot** and point out high-RUL compression.
5. **Show the error-by-bin plot** and explain the 30–90 late-rate risk.
6. **Show the ablation** and say the features, not the model, drive the gain.
7. **Open the Streamlit app**:
   - select an engine,
   - show predicted RUL, priority, actual RUL, and late warning,
   - switch to batch predictions and download the CSV,
   - open the model/metrics page.
8. **Close with limitations**: simulated data, one condition, no uncertainty,
   decision-support only.

---

## 7. Short answers for quick revision

- **Task:** RUL regression, not classification.
- **Dataset:** NASA C-MAPSS FD001, simulated turbofan.
- **Target:** cycles to failure, capped at 125.
- **Split:** 80/20 by engine; 5 truncations per validation engine.
- **Features:** 213; differences + rolling stats over 10/30/50.
- **Models:** mean baseline, Ridge, RF, HistGradientBoosting, XGBoost.
- **Selected:** Ridge, α = 1.0.
- **Test RMSE:** 14.89 cycles (capped), 16.30 (raw).
- **Test MAE:** 11.63 cycles (capped).
- **Test R²:** 0.862.
- **NASA score:** 3.20 (capped).
- **Late rate:** 50/100.
- **Ablation:** engineered features cut RMSE 27.7% and NASA 74.0% for Ridge.
- **Prototype:** Streamlit, four views, CSV upload/download.
- **Main limitation:** simulated data, one operating condition.
- **Ethics:** decision-support only, human oversight, asymmetric risk.

---

## 8. Presentation outline (10 slides)

| Slide | Title | Key content |
|---:|---|---|
| 1 | Title | Project 46, RUL estimation, C-MAPSS, name/roll number |
| 2 | Motivation | Corrective vs preventive vs predictive maintenance; cost of late prediction |
| 3 | Problem statement | RUL definition, capped target at 125, regression not classification |
| 4 | Dataset | FD001: 100/100 engines, 20,631/13,096 rows, 21 sensors, 6 constant |
| 5 | Methodology | Engine-level split, 5 truncations per validation engine, 213 features, 5 models |
| 6 | Validation results | Model comparison table; Ridge selected; baseline context |
| 7 | Test results | RMSE 14.89, MAE 11.63, R² 0.862, NASA 3.20; predicted-vs-true plot |
| 8 | Error analysis and ablation | Error by RUL range; worst cases; 27.7% RMSE and 74.0% NASA improvement |
| 9 | Prototype | Engine demo, batch priorities, CSV upload, planning buffer |
| 10 | Limitations and conclusion | Simulated data, one condition, no uncertainty, human oversight, future work |

**Demo timing:** 20 seconds on the engine demo, 20 seconds on batch
priorities, 10 seconds on the model/metrics page. Keep the rest of the
presentation under eight minutes.
