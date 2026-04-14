# MetAIsAFe — Machine Learning Pipeline: Technical Description and Methodological Rationale

**Project:** MetAIsAFe — Meta-modelling of Hi-sAFe Agroforestry Simulations  
**Author:** Étienne SABY  
**Version:** 2.0 (pipeline B2)  
**Date:** 2026-05  

---

## Abstract

MetAIsAFe is a surrogate modelling framework designed to emulate the behaviour of Hi-sAFe, a mechanistic three-dimensional model of temperate agroforestry systems developed by INRAE. Because a single 40-year Hi-sAFe simulation requires tens of minutes of CPU time, systematic exploration of the input parameter space — essential for global sensitivity analysis, climate scenario comparison, and decision-support applications — is computationally prohibitive at scale. MetAIsAFe replaces the mechanistic model with a lightweight, data-driven surrogate that predicts key agroforestry performance indicators in milliseconds, enabling the evaluation of thousands of scenarios that would otherwise require weeks of high-performance computing.

The pipeline is implemented in Python and structured into nine sequential steps, from raw simulation output ingestion to post-processed ratio computation. It is designed around three core principles: strict prevention of data leakage, physical consistency of predictions, and modularity for long-term maintainability.

---

## 1. Scientific Context and Modelling Strategy

### 1.1 The Hi-sAFe Simulation Corpus

The training dataset consists of approximately 2,048 Hi-sAFe simulations generated via a Sobol quasi-random sampling plan (Saltelli et al., 2010), covering the pedo-climatic and structural parameter space representative of temperate and Mediterranean agroforestry systems in France and north-western Europe. Each simulation spans 40 harvest years and produces daily outputs that are aggregated into per-cycle indicators: grain yield, aboveground biomass, stem carbon stocks, and their respective sole-crop or sole-forest references.

The resulting meta-table contains one row per harvest cycle per simulation (approximately 40–80 rows per SimID depending on crop rotation), with columns representing both Sobol plan parameters (inputs) and aggregated Hi-sAFe outputs (targets).

### 1.2 Two-Level Prediction Architecture

A fundamental design decision of MetAIsAFe is to separate prediction into two levels:

**Level 1 — Machine learning of physical stocks.**  
The surrogate directly predicts four physical stock variables: `carbonStem_AF`, `carbonStem_TF`, `yield_AF`, and `yield_TA`. These variables are always non-negative, physically continuous, and well-defined across all simulation phases.

**Level 2 — Analytical computation of performance ratios.**  
The Land Equivalent Ratio (LER) and Response Ratio (RR) — the primary agroforestry performance indicators — are computed analytically from the predicted stocks via the classical formulations (Mead & Willey, 1980):

$$\text{RR}_\text{crop} = \frac{\text{yield}_\text{eff,AF}}{\text{yield}_\text{eff,TA}}, \quad \text{LER} = \text{RR}_\text{crop} + \text{RR}_\text{tree}$$

where `_eff_` denotes area-corrected effective stocks accounting for tree strip geometry.

**Rationale for this separation.**  
Direct prediction of ratios was tested during preliminary experiments and systematically rejected for three reasons. First, during the juvenile tree phase (years 1–10), tree carbon stocks approach zero, causing ratio denominators to be near-zero and producing numerically unstable training targets with extreme values. Second, interpolating ratios to fill the resulting NaN values produces physically meaningless trajectories and leads to severe overfitting (R² < 0 on the test set for several targets). Third, the analytical approach provides controlled error propagation: the uncertainty on predicted ratios scales approximately as twice the uncertainty on individual stock predictions, and remains physically bounded.

---

## 2. Pipeline Architecture

The pipeline is organised into nine sequential steps managed by dedicated Python modules. The orchestration layer (`pipeline.py`) chains these steps for a given training campaign.

```
RAW DATA
   │
   ▼
[STEP 1]  Load & Prepare          data/loader.py, data/preparation.py
   │
   ▼
[STEP 2]  Population Filtering    data/preparation.py
   │
   ▼
[STEP 3]  Train/Test Split + CV   data/splitter.py
   │
   ▼
[STEP 4]  Encoding & Winsorisation data/loader.py, data/preprocessing.py
   │
   ▼
[STEP 5a] Cascade Classifiers     modeling/classifiers.py, modeling/trainer.py
   │
   ▼
[STEP 5b] Stage 1 Regressors      modeling/models.py, modeling/trainer.py
   │       carbonStem_AF, carbonStem_TF
   ▼
[STEP 5c] Stage 2 Regressors      modeling/models.py, modeling/trainer.py
   │       yield_AF, yield_TA (+ Stage 1 predictions as features)
   ▼
[STEP 6]  SHAP Analysis           modeling/shap_analysis.py
   │
   ▼
[STEP 7]  Evaluation              modeling/evaluator.py
   │
   ▼
[STEP 8]  Inference               modeling/predictor.py
   │
   ▼
[STEP 9]  Post-processing         data/preprocessing.py
           RR, LER (analytical)
```

---

## 3. Step-by-Step Description

### Step 1 — Data Loading and Preparation (`data/loader.py`, `data/preparation.py`)

Raw simulation outputs are stored as Parquet files (preferred for dtype preservation and read performance) or CSV. Loading is handled by `load_data()`, which supports optional Polars acceleration for large files.

Preparation applies four sequential transformations to the raw meta-table:

1. **`add_derived_columns()`** — Creates `Harvest_Year_Absolute`, a per-SimID normalised temporal index (cycle 1 → N), which is the primary temporal feature used by all regressors. A Boolean `Rotation` flag is also derived from the `rot_id` column.

2. **`filter_crops()`** — Removes rows corresponding to excluded crop species (e.g. oilseed rape, which was excluded from Batch 2 due to insufficient sample size). Reports the number of SimIDs completely lost.

3. **`clean()`** — Conservative cleaning: drops fully-NA columns and exact duplicate rows; clips stress fraction variables to [0, 1] and physical stocks to ≥ 0; warns about columns with high NA rates (> 30%) and duplicate (SimID, Cycle, Crop) keys. No simulation is excluded at this stage.

4. **`compute_effective_vars()`** — Computes area-corrected effective stocks (`_eff_AF`, `_eff_TA`, `_eff_TF`) from raw stocks and plot geometry parameters. These are used exclusively in post-processing (Step 9) and are never passed as ML features or targets.

### Step 2 — Population Filtering (`data/preparation.py`)

`filter_population()` classifies each SimID into one of four structural populations based on its simulation trajectory, and optionally filters the dataset to a single target population.

**Classification rules:**

- **Tree failure** (`tree_failed`): `carbonStem_AF` at the last observed cycle < 1.0 kgC/tree. This threshold identifies simulations where the agroforestry tree component failed to establish viable growth over 40 years.

- **Yield failure** (`yield_failed`): the fraction of cycles where either `yield_AF` or `yield_TA` falls below 0.5 t/ha exceeds 50%. This joint criterion captures geographic/climatic crop failure, which is highly co-occurring between the AF and reference systems (88.7% co-occurrence in B2 analysis).

The four resulting populations and their approximate frequencies in Batch 2 are:

| Population | Description | B2 frequency |
|---|---|---|
| `yield_ok × tree_ok` | Main meta-model training set | 46.7% |
| `yield_ok × tree_failed` | Cultural-only model | 36.9% |
| `yield_fail × tree_ok` | Geographic rejection | 7.5% |
| `yield_fail × tree_failed` | Full rejection | 9.0% |

The main pipeline trains on the `yield_ok × tree_ok` population (954 SimIDs in B2). The cascade classifiers trained in Step 5a are responsible for routing new simulations to the appropriate population at inference time.

### Step 3 — Train/Test Split and Cross-Validation Setup (`data/splitter.py`)

**Core constraint: the split unit is always the SimID, never the individual row.**  
Each SimID is a time series of harvest cycles. A row-level random split would allow the model to observe cycle 15 during training and predict cycle 14 during evaluation, constituting a severe temporal data leakage. All split functions in `splitter.py` enforce SimID-level integrity.

`stratified_split_by_rotation()` is the recommended split function for Batch 2. It uses rotation signatures (the alphabetically sorted list of unique crops per SimID) to ensure that monoculture and rotation SimIDs appear in the same proportions in both train and test sets. The default split ratio is 80%/20%.

For cross-validation, `make_group_kfold()` returns a `GroupKFold(k=5)` splitter, and `build_cv_groups()` extracts the SimID column as the group array. This ensures that all cycles of a given simulation remain in the same fold throughout cross-validation.

### Step 4 — Encoding and Winsorisation (`data/loader.py`, `data/preprocessing.py`)

**Both operations are applied after the train/test split to prevent data leakage.**

`encode_categoricals()` supports two strategies:
- **LightGBM mode** (default): categorical columns (`main_crop`, `w_type`) are cast to pandas `category` dtype, which LightGBM handles natively via histogram-based split finding. No fitting is involved, making this transformation truly leakage-free.
- **Sklearn mode**: `LabelEncoder` is fitted on the training set only and applied to the test set. Required for CART surrogate fitting; not used in the main LightGBM pipeline.

`apply_winsorization()` caps extreme outlier values in the four stock target columns at the [1st, 99th] percentile bounds. Bounds are computed exclusively from the training set and applied to both splits, following the standard fit/transform protocol of sklearn preprocessing.

### Step 5a — Cascade Classifiers (`modeling/classifiers.py`, `modeling/trainer.py`)

Before training the regression models, two binary classifiers are fitted on the **full dataset** (all four populations) to learn the structural boundaries between populations. They are trained at SimID level (one feature vector per simulation, aggregated as the first row, since all Sobol parameters are constant within a SimID).

**CLF1 — Tree Failure Classifier.**  
Predicts whether a simulation will result in tree failure (`carbonStem_AF < 1.0 kgC/tree` at final cycle). The failure mechanism is primarily pedological: clay content, stone fraction, and sand fraction dominate feature importance (> 60% cumulated). Geographic coordinates modulate the pedological signal via their correlation with climate stress. LightGBM is used with features: `clay, sand, stone, latitude, longitude, plotHeight, plotWidth, soilDepth, main_crop`. Accuracy on B2: ~66.7% (the boundary is intentionally soft — the failure zone is diffuse in soil texture space).

**CLF2 — Yield Failure Classifier.**  
Predicts whether a simulation results in systematic crop failure. The failure mechanism is primarily geographic/climatic: latitude is the dominant feature (44.5–53.0% importance), followed by longitude. LightGBM features: `latitude, longitude, clay, waterTable, sand, stone, soilDepth`. Accuracy on B2: ~91.5% (sharp geographic boundary).

**Geographic fallback rule.**  
If the minority class in the CLF2 training data contains fewer than 80 samples (insufficient for reliable LightGBM training), a deterministic rule derived from B1/B2 decision tree analysis is applied:

$$\text{yield\_failed} = (\text{latitude} < 44.0°\text{N}) \cap (\text{longitude} > 3.5°\text{E})$$

This rule achieves ~90% accuracy and avoids overfitting on highly imbalanced datasets.

### Step 5b — Stage 1 Regression: Tree Carbon Stocks (`modeling/models.py`, `modeling/trainer.py`)

Two LightGBM regressors are trained on the `yield_ok × tree_ok` population to predict tree carbon stocks:

- `carbonStem_AF`: stem carbon in the agroforestry system (kgC/tree)
- `carbonStem_TF`: stem carbon in the sole-forest reference (kgC/tree)

Tree stocks are predicted first because they exhibit a smoother, more monotonic temporal trajectory (cumulative growth), making them more amenable to accurate surrogate modelling. Their predictions are then used as additional features for the Stage 2 crop yield regressors.

Each model is evaluated via 5-fold GroupKFold cross-validation before final training on the full training set.

### Step 5c — Stage 2 Regression: Crop Yields (`modeling/models.py`, `modeling/trainer.py`)

Two LightGBM regressors predict:

- `yield_AF`: grain yield in the agroforestry system (t/ha per cell)
- `yield_TA`: grain yield in the sole-crop reference (t/ha)

**Sequential (cascade) design rationale.**  
Stage 2 models receive, as additional input features, the predictions of `carbonStem_AF` and `carbonStem_TF` from Stage 1. This design is motivated by the strong physical coupling between tree growth and crop yield in agroforestry: tree canopy expansion creates light competition that progressively reduces crop yield, while tree root competition for water and nitrogen creates further interactions. Incorporating tree carbon predictions as features allows Stage 2 models to explicitly represent this competition-mediated relationship without requiring the full mechanistic simulation.

The `extra_train_features` parameter of `cross_validate()` handles the injection of Stage 1 predictions during cross-validation, concatenating them with the standard feature matrix for each fold.

### Step 6 — SHAP Analysis (`modeling/shap_analysis.py`)

SHAP (SHapley Additive exPlanations) values are computed for all four trained regressors using `shap.TreeExplainer`, which is exact and computationally efficient for tree-based models (Lundberg & Lee, 2017).

Three outputs are produced per model:
- Raw SHAP value matrix (n_samples × n_features)
- Mean absolute SHAP summary (`mean(|SHAP|)` per feature, equivalent to mean absolute feature contribution)
- Beeswarm visualisation (`plot_shap_by_target()`)

An export function (`export_shap_for_shiny()`) produces RShiny-compatible CSV files for integration into the decision-support interface.

### Step 7 — Evaluation (`modeling/evaluator.py`)

All models are evaluated on the held-out test set using a comprehensive set of regression metrics:

| Metric | Formula | Interpretation |
|---|---|---|
| R² | $1 - \frac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2}$ | Variance explained |
| RMSE | $\sqrt{\frac{1}{n}\sum(y-\hat{y})^2}$ | Absolute error scale |
| MAE | $\frac{1}{n}\sum|y-\hat{y}|$ | Robust to outliers |
| MAPE | $\frac{1}{n}\sum\frac{|y-\hat{y}|}{|y|}$ | Relative error |
| NRMSE | RMSE / (max − min) | Scale-normalised error |
| Bias | $\frac{1}{n}\sum(\hat{y}-y)$ | Systematic over/under-prediction |
| Pearson r | — | Linear correlation |

Diagnostic plots (observed vs. predicted scatter, residuals vs. predicted, residual distribution) are generated for each target and saved to the campaign `Plots/Diagnostics/` directory.

### Step 8 — Inference (`modeling/predictor.py`)

`build_inference_grid()` generates the 40-row inference matrix (one row per harvest year) from a user-provided parameter dictionary. This replicates the temporal structure of a Hi-sAFe simulation without running the mechanistic model.

`predict_cascade()` applies the full two-stage routing logic:
1. CLF1 predicts tree failure → routes to zero carbon stocks if failed
2. CLF2 predicts yield failure (or geographic rule) → routes to zero yields if failed
3. If both are OK: Stage 1 and Stage 2 regressors predict stocks

`predict_single_sim()` wraps the full inference chain for a single parameter set, returning a 40-year time series of all predicted stocks and derived ratios.

### Step 9 — Post-processing (`data/preprocessing.py`)

`compute_ratios_from_stocks()` computes RR and LER analytically from the predicted `_eff_` stocks. Division by zero is guarded with a configurable threshold (default: 0.001), below which the ratio is set to NaN to avoid numerical instability in the juvenile phase.

---

## 4. Model Selection Rationale

### 4.1 LightGBM as Primary Model

Following preliminary benchmarking across all four stock targets using 5-fold GroupKFold cross-validation, LightGBM (Ke et al., 2017) was selected as the sole production model. Mean validation R² scores showed LightGBM outperforming Random Forest by 4–9 percentage points and XGBoost by 1–3 points, with training time reduced by a factor of 3–5× relative to XGBoost and up to 15× relative to Random Forest.

This result is consistent with the broader literature on gradient-boosted trees applied to structured simulation data: LightGBM's leaf-wise tree growth strategy and histogram-based split finding are particularly well-suited to tabular datasets with moderate feature counts, mixed numeric-categorical inputs, and high signal-to-noise ratio — all characteristics of the Hi-sAFe meta-table. Native `category` dtype support eliminates the need for explicit label encoding and the associated leakage risk.

XGBoost is maintained as an optional fallback for validation purposes. Random Forest is retired from the main pipeline.

### 4.2 Hyperparameter Configuration

Production hyperparameters were set by expert initialisation and validated via cross-validation:

```
n_estimators    : 500
learning_rate   : 0.05
num_leaves      : 63
max_depth       : -1 (unlimited — leaf-wise growth)
min_child_samples: 20
subsample       : 0.8
colsample_bytree: 0.8
reg_alpha       : 0.1
reg_lambda      : 0.1
importance_type : "gain"  (required for SHAP TreeExplainer compatibility)
```

Bayesian hyperparameter optimisation via Optuna (Akiba et al., 2019) is available through `trainer.tune_optuna()` with a TPE sampler, using GroupKFold mean R² as the objective. It is not applied by default in the production pipeline to preserve reproducibility.

---

## 5. Data Leakage Prevention

Data leakage is the primary risk in surrogate modelling of time-series simulation data. The MetAIsAFe pipeline implements leakage prevention at four levels:

| Risk | Mitigation |
|---|---|
| Temporal leakage (cycle-level) | Split unit = SimID; GroupKFold by SimID |
| Preprocessing leakage | Winsorisation bounds fitted on train only |
| Encoding leakage | LightGBM `category` dtype (no fit); sklearn `LabelEncoder` fitted on train only |
| Target leakage | Ratios (RR, LER) computed post-inference only; `_eff_` columns excluded from feature sets by taxonomy |

---

## 6. Column Taxonomy and Feature Engineering

`column_taxonomy.py` serves as the single source of truth for all column classifications. Key feature groups for Batch 2 (`ACTIVE_FEATURES_B2`) are:

| Group | Features | N |
|---|---|---|
| Temporal | `Harvest_Year_Absolute` | 1 |
| Geographic | `latitude`, `longitude` | 2 |
| Plot design | `plotWidth`, `plotHeight` | 2 |
| Soil | `soilDepth`, `sand`, `clay`, `stone`, `waterTable` | 5 |
| Categorical | `main_crop`, `w_type` | 2 |

Features excluded from B2 (zero variance — fixed in Sobol plan): `strip_width` (= 3), `northOrientation` (= 90°), `Rotation` (= FALSE), `period` (= FUT), `w_amp`, `w_mean`, `w_peak_doy`.

Climate variables (`GDD_cycle_AF`, `ETP_cycle_AF`, etc.) are excluded from B2 features because they are Hi-sAFe outputs, not Sobol plan inputs, and are therefore unavailable at inference time for new parameter combinations.

---

## 7. Limitations and Perspectives

**Interpolation scope.** Linear within-SimID interpolation of tree carbon stocks (`interpolate_dynamic_vars()`) is available but not applied in the main B2 pipeline. It is restricted to physically monotone stocks (tree carbon compartments) and explicitly excluded for crop yields, deltas, and ratio variables.

**Classifier accuracy.** CLF1 (tree failure) achieves only ~66.7% accuracy, reflecting the genuinely diffuse boundary of the failure zone in soil texture space. Misclassification of a tree-failed simulation as tree-ok will produce non-zero carbon stock predictions. This is an acceptable trade-off given the low frequency of the failure class (16.5% in B2) and the soft physical interpretation of the threshold.

**Sequential bias.** The Stage 2 regressors are conditioned on Stage 1 predictions, meaning that errors in `carbonStem_AF` prediction propagate to `yield_AF` and `yield_TA`. Error propagation analysis (σ(yield) as a function of σ(carbonStem)) is recommended as part of the uncertainty quantification work.

**Climate scenarios.** The current B2 pipeline is trained on the FUT climate scenario exclusively. Extension to PRE scenarios and scenario-conditional prediction requires reintegrating `period` as an active feature, which is straightforward given the existing `column_taxonomy` structure.

---

## 8. Software Dependencies

| Package | Version tested | Role |
|---|---|---|
| Python | ≥ 3.9 | Runtime (`Path.with_stem` requires 3.9+) |
| lightgbm | ≥ 4.0 | Primary model |
| scikit-learn | ≥ 1.3 | Splitters, metrics, preprocessing |
| shap | ≥ 0.44 | TreeExplainer |
| optuna | ≥ 3.0 | Bayesian optimisation |
| pandas | ≥ 2.0 | Data manipulation |
| numpy | ≥ 1.24 | Numerical operations |
| joblib | ≥ 1.3 | Model serialisation |
| pyarrow | ≥ 14.0 | Parquet I/O |
| polars | optional | Fast loading for large files |
| pyreadr | optional | FST format (R binary) |
| tqdm | ≥ 4.0 | Cross-validation progress bar |

---

## References

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD*, 2623–2631.
- Bentéjac, C., Csörgő, A., & Martínez-Muñoz, G. (2021). A comparative analysis of gradient boosting algorithms. *Artificial Intelligence Review*, 54, 1937–1967.
- de Coligny, F., et al. (2020). Hi-sAFe: A 3D Biophysical Model for Agroforestry Systems. *Sustainability*, 12(5), 2149.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*, 30.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30.
- Mead, R., & Willey, R. W. (1980). The concept of a land equivalent ratio and advantages in yields from intercropping. *Experimental Agriculture*, 16(3), 217–228.
- Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S. (2010). Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index. *Computer Physics Communications*, 181(2), 259–270.