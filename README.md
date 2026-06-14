# Groundwater Missing-Value Imputation Code

The workflow contains three modules:

1. **M1: Bi-LSTM sequence module**  
Predicts missing values using the target well's own observations before and after a missing segment.
2. **M2: XGBoost external-feature module**  
Predicts the target well using other wells and user-provided external predictors.
3. **Gate: neural fusion module**  
Learns the fusion weight between M1 and M2.

The code is designed to support both first-time model training and later imputation using saved models.

\---

## 1\. Files

```text
groundwater\\\\\\\_imputation\\\\\\\_first\\\\\\\_run.py              # First-time training + imputation
groundwater\\\\\\\_imputation\\\\\\\_apply\\\\\\\_saved\\\\\\\_models.py     # Imputation using already trained models
groundwater\\\\\\\_imputation\\\\\\\_models.py                 # Core models, utilities, training, and prediction functions
README.md                                        # This file
```

\---

## 2\. Input Data Format

The input file should be an Excel file containing one date column, several groundwater well columns, and optional external predictor columns.

Example:

```text
date | well\\\\\\\_1 | well\\\\\\\_2 | well\\\\\\\_3 | rainfall | temperature | evaporation
```

Column-detection rules:

* The date column must be named `date` or `Date`.
* Groundwater well columns must contain the string `well` in their names.
* If `--external-cols` is not specified, all non-date and non-well columns are treated as external predictors.

\---

## 3\. Note on the Provided Dataset

The accompanying dataset serves as the validation dataset in this study and was obtained from the United States Geological Survey (USGS) website (https://www.usgs.gov/).

This dataset is a complete groundwater dataset used for model training and performance evaluation. It is equivalent to the selected complete training dataset, i.e., `data.xlsx`, rather than a raw `dataall.xlsx` file containing real missing groundwater segments.

If the input dataset contains no missing well values, the code will still train and evaluate the models using simulated missing segments. In this case, no real values need to be imputed, and the final imputed dataset may be identical to the input. The main output of interest is then:

```text
model\_performance\_summary.xlsx
```

To test real missing-value filling, users should provide a groundwater dataset containing missing values in one or more well columns.

\---

## 4\. Installation

Recommended Python version: 3.9--3.11.

Install the required packages:

```bash
pip install numpy pandas openpyxl scikit-learn scipy tensorflow torch xgboost matplotlib
```

A GPU is optional. The code can run on CPU, although model training may take longer.

\---

## 5\. First-Time Training and Imputation

Run:

```bash
python groundwater\\\\\\\_imputation\\\\\\\_first\\\\\\\_run.py --input dataall.xlsx --output dataall\\\\\\\_imputed.xlsx --results-dir results
```

This command will:

1. Read the input Excel file.
2. Select the longest complete row-contiguous sequence for model training.
3. Save the selected sequence as `results/data.xlsx`.
4. Train M1, M2, and the gating network for each target well.
5. Save all deployable models and manifests under `results/models/`.
6. Produce an overall model-performance summary.
7. Return to the original input file and impute missing well values if missing values exist.

\---

## 6\. Imputation Using Saved Models

After models have been trained once, a new file can be imputed without retraining:

```bash
python groundwater\\\\\\\_imputation\\\\\\\_apply\\\\\\\_saved\\\\\\\_models.py --input new\\\\\\\_dataall.xlsx --output new\\\\\\\_dataall\\\\\\\_imputed.xlsx --models-dir results/models
```

The new file should use the same column names and predictor structure as the training data.

If the default names are used, the script can also be run directly:

```bash
python groundwater\\\\\\\_imputation\\\\\\\_apply\\\\\\\_saved\\\\\\\_models.py
```

Default settings:

```text
Input file:    dataall.xlsx
Output file:   dataall\\\\\\\_imputed\\\\\\\_by\\\\\\\_saved\\\\\\\_models.xlsx
Models folder: results/models
```

Do not delete `results/models/` if future imputation without retraining is required.

\---

## 7\. Main Outputs

After first-time training, the output directory contains:

```text
results/
├── data.xlsx
├── dataall\\\\\\\_imputed.xlsx
├── model\\\\\\\_performance\\\\\\\_summary.xlsx
├── training\\\\\\\_block\\\\\\\_summary.json
└── models/
```

### `data.xlsx`

The automatically selected longest complete training sequence.

### `dataall\\\\\\\_imputed.xlsx`

The final output dataset. It contains two sheets:

* `imputed\\\\\\\_data`: the original data table with missing well values filled where possible.
* `imputation\\\\\\\_log`: one row per processed missing point, including the target well, date, prediction method, model predictions, fusion weights, and final imputed value.

### `model\\\\\\\_performance\\\\\\\_summary.xlsx`

A compact performance summary evaluated on the selected complete dataset.

It contains:

* `selected\\\\\\\_training\\\\\\\_block`: metadata of the selected complete training sequence.
* `overall\\\\\\\_metrics`: overall train/validation/test metrics for M1, M2, and the fused model.

Reported metrics include:

```text
RMSE, NSE, MAE, MAPE
```

### `training\\\\\\\_block\\\\\\\_summary.json`

A short summary of the selected complete training block, including row number, start date, end date, well columns, external predictor columns, and split settings.

### `models/`

Saved deployable models and scalers. Each well folder contains a `model\\\\\\\_manifest.json` file that records model paths, feature order, scalers, trained M1 step lengths, and gating input order.

The manifest files must not be deleted if direct imputation using saved models is needed.

\---

## 8\. Missing-Gap Handling Rule

For each target well, the code scans continuous missing segments in the original dataset.

Let the real missing length be `L`.

* If M1 has enough observed target-well values before and after the missing segment, M1 and M2 are fused by the gating network.
* If M1 context is unavailable, for example at the beginning or end of the record, M2-only imputation is used.
* If `L` is longer than the maximum trained M1 step, the first `max\\\\\\\_step` points are imputed using gated fusion, and the remaining tail is imputed by M2 only.
* Newly imputed values are not recursively used as predictors for other missing points.

\---

## 9\. Important Notes

The default parameter settings are provided for the example implementation and should not be regarded as universally optimal.

The following settings should be adjusted according to the actual dataset:

```text
CONTEXT\\\\\\\_LENGTH      # Temporal window used by the Bi-LSTM module
TRAIN\\\\\\\_END\\\\\\\_IDX      # Training/validation split boundary
VAL\\\\\\\_END\\\\\\\_IDX        # Validation/test split boundary
TS\\\\\\\_FORECAST\\\\\\\_STEPS  # Missing lengths used to train M1
Hyperparameter ranges for Bi-LSTM, XGBoost, and the gating network
```

These settings should be selected based on the temporal resolution, record length, missing-gap durations, data availability, and computational resources.

The selected complete training block should preferably cover a full seasonal or hydrological cycle. If the selected block is too short or does not represent the variability of the full dataset, model generalization may be limited, especially for boundary gaps where M1 cannot be used and the imputation relies mainly on M2.

Users are encouraged to inspect:

```text
training\\\\\\\_block\\\\\\\_summary.json
model\\\\\\\_performance\\\\\\\_summary.xlsx
```

before interpreting the imputation results.

\---

## 10\. Reproducibility Notes

The code sets random seeds where possible. However, exact reproducibility may still be affected by hardware, operating system, TensorFlow/PyTorch backend behavior, XGBoost version, and floating-point computation order.

Small numerical differences between runs are possible.

