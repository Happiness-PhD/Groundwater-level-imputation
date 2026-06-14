# -*- coding: utf-8 -*-
"""Impute groundwater missing values using previously trained models.

This script is designed for the second-stage use case:
models have already been trained by groundwater_imputation_first_run.py,
and a new or existing dataall.xlsx file needs to be imputed directly
without retraining.

Default behavior
----------------
Input file:
    dataall.xlsx

Output file:
    dataall_imputed_by_saved_models.xlsx

Model directory:
    results/models
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from groundwater_imputation_models import run_saved_model_imputation


def parse_external_cols(text: Optional[str]) -> Optional[list[str]]:
    """Parse comma-separated external predictor column names."""
    if text is None or not str(text).strip():
        return None
    return [x.strip() for x in str(text).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply saved groundwater imputation models to dataall.xlsx."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="dataall.xlsx",
        help="Input Excel file with missing well values. Default: dataall.xlsx",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataall_imputed_by_saved_models.xlsx",
        help="Output imputed Excel file. Default: dataall_imputed_by_saved_models.xlsx",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="results/models",
        help="Directory containing <well>/model_manifest.json files. Default: results/models",
    )

    parser.add_argument(
        "--external-cols",
        type=str,
        default=None,
        help="Optional comma-separated external predictor columns.",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    models_dir = Path(args.models_dir).resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Please place dataall.xlsx in the current working directory, "
            "or specify another file using --input."
        )

    if not models_dir.exists():
        raise FileNotFoundError(
            f"Models directory not found: {models_dir}\n"
            "Please run groundwater_imputation_first_run.py first, "
            "or specify the correct model directory using --models-dir."
        )

    print("=" * 80)
    print("Groundwater imputation using saved models")
    print("=" * 80)
    print(f"Input file : {input_path}")
    print(f"Output file: {output_path}")
    print(f"Models dir : {models_dir}")
    print("=" * 80)

    run_saved_model_imputation(
        input_file=str(input_path),
        output_file=str(output_path),
        models_dir=str(models_dir),
        external_cols=parse_external_cols(args.external_cols),
    )

    print("=" * 80)
    print(f"Imputation completed. Output saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()