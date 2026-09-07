Groundwater-Level Imputation Code

This repository provides three Python scripts for model training and groundwater-level imputation. Input datasets and pretrained models are not included. Users must prepare their own data and train the models before applying saved models.

1. Files

groundwater_imputation_first_run.py: Train models and impute the input data.

groundwater_imputation_apply_saved_models.py: Impute data using saved models.

groundwater_imputation_models.py: Core training, saving, loading, and prediction functions.

2. Input Requirements

Prepare an Excel file, named dataall.xlsx by default.

The date column must be named date or Date.

Well-column names must contain well, for example well_1 and well_2.

All other columns are treated as external predictors unless --external-cols is specified.

With the default settings, first-time training requires at least 3001 consecutive rows with all modelling variables available.

3. Installation

Python 3.10 or later is required.

Install the dependencies:

pip install numpy pandas openpyxl scikit-learn scipy tensorflow torch xgboost matplotlib

4. Running the Code

Run these commands from the directory containing the three scripts and your input file.

First-time training and imputation:

python groundwater_imputation_first_run.py --input dataall.xlsx --output dataall_imputed.xlsx --results-dir results

After training, impute data using the saved models:

python groundwater_imputation_apply_saved_models.py --input dataall.xlsx --output dataall_imputed_by_saved_models.xlsx --models-dir results/models

Use --input to specify another Excel file. For saved-model imputation, use the same well-column names and predictor structure as the training data.

5. Main Outputs

results/dataall_imputed.xlsx: Data imputed by the first-run script, with imputed_data and imputation_log worksheets.

dataall_imputed_by_saved_models.xlsx: Data imputed by the saved-model script, with the same two worksheets.

results/model_performance_summary.xlsx: Model performance summary.

results/data.xlsx and results/training_block_summary.json: Selected training data and selection details.

results/models/: Saved models, scalers, and model manifests. Keep this directory for future imputation without retraining.
