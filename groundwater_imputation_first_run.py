# -*- coding: utf-8 -*-
"""First-time training and imputation pipeline for the groundwater imputation model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from groundwater_imputation_models import (
    TRAIN_END_IDX,
    VAL_END_IDX,
    clean_column_names,
    collect_overall_performance,
    cleanup_intermediate_files,
    identify_columns,
    infer_time_delta,
    run_saved_model_imputation,
    select_longest_complete_block,
    train_all_wells,
    write_json,
)


def parse_external_cols(text: Optional[str]) -> Optional[list[str]]:
    if text is None or not str(text).strip():
        return None
    return [x.strip() for x in str(text).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the groundwater imputation models and impute missing values in one command."
    )
    parser.add_argument("--input", type=str, default="dataall.xlsx", help="Original Excel file containing missing values.")
    parser.add_argument("--output", type=str, default="dataall_imputed.xlsx", help="Name of the final imputed Excel file.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for outputs and saved models.")
    parser.add_argument("--external-cols", type=str, default=None, help="Optional comma-separated external predictor columns.")
    parser.add_argument("--min-complete-rows", type=int, default=3001, help="Minimum length of the selected complete training sequence.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately if a target-well training process fails.")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep detailed temporary CSV files generated during training.")
    parser.add_argument("--no-sort-by-date", action="store_true", help="Do not sort by date before selecting the complete training block.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    output_arg = Path(args.output)
    output_path = output_arg if output_arg.is_absolute() else results_dir / output_arg
    output_path = output_path.resolve()

    print("=" * 88)
    print("Groundwater imputation pipeline")
    print(f"Input file:   {input_path}")
    print(f"Results dir:  {results_dir}")
    print(f"Models dir:   {models_dir}")
    print(f"Output file:  {output_path}")
    print("=" * 88)

    print("\n[1/5] Selecting the longest complete training sequence...")
    df_all = clean_column_names(pd.read_excel(input_path))
    external_cols = parse_external_cols(args.external_cols)
    col_info = identify_columns(df_all, external_cols=external_cols)
    required_cols = col_info.well_cols + col_info.external_cols

    print(f"Date column: {col_info.date_col}")
    print(f"Well columns: {col_info.well_cols}")
    print(f"External columns: {col_info.external_cols}")

    start_idx, end_idx, block, _segments = select_longest_complete_block(
        df=df_all,
        date_col=col_info.date_col,
        required_cols=required_cols,
        sort_by_date=not args.no_sort_by_date,
    )

    if len(block) < args.min_complete_rows:
        raise ValueError(
            f"The longest complete block has only {len(block)} rows, smaller than "
            f"min_complete_rows={args.min_complete_rows}. The model uses TRAIN_END_IDX={TRAIN_END_IDX} "
            f"and VAL_END_IDX={VAL_END_IDX}; therefore at least 3001 rows are recommended."
        )

    modelling_cols = [col_info.date_col] + col_info.well_cols + col_info.external_cols
    block = block[modelling_cols].copy()
    data_path = results_dir / "data.xlsx"
    block.to_excel(data_path, index=False)

    training_summary = {
        "input_file": str(input_path),
        "generated_training_file": str(data_path),
        "selected_start_index_in_sorted_dataall": int(start_idx),
        "selected_end_index_in_sorted_dataall": int(end_idx),
        "selected_rows": int(len(block)),
        "selected_start_date": str(block[col_info.date_col].iloc[0]),
        "selected_end_date": str(block[col_info.date_col].iloc[-1]),
        "inferred_time_delta": str(infer_time_delta(block[col_info.date_col])),
        "date_col": col_info.date_col,
        "well_cols": col_info.well_cols,
        "external_cols": col_info.external_cols,
        "train_end_idx": TRAIN_END_IDX,
        "val_end_idx": VAL_END_IDX,
        "selection_rule": "Longest row-contiguous block where all well and external predictor columns are non-missing; time-interval continuity is not enforced.",
    }
    summary_json = results_dir / "training_block_summary.json"
    write_json(training_summary, str(summary_json))
    print(f"Selected block: {training_summary['selected_start_date']} -> {training_summary['selected_end_date']} ({len(block)} rows)")
    print(f"Saved complete training dataset: {data_path}")

    print("\n[2/5] Training well-specific M1, M2, and gating models...")
    train_all_wells(data_file=str(data_path), models_dir=str(models_dir), stop_on_error=args.stop_on_error)

    print("\n[3/5] Writing compact performance summary...")
    performance_xlsx = results_dir / "model_performance_summary.xlsx"
    collect_overall_performance(
        models_dir=str(models_dir),
        training_block_summary_json=str(summary_json),
        output_xlsx=str(performance_xlsx),
    )
    print(f"Saved performance summary: {performance_xlsx}")

    if not args.keep_intermediate:
        print("\n[4/5] Removing non-essential intermediate files while keeping saved models...")
        cleanup_intermediate_files(str(models_dir))
    else:
        print("\n[4/5] Keeping intermediate files as requested.")

    print("\n[5/5] Imputing the original dataset using the saved models...")
    run_saved_model_imputation(
        input_file=str(input_path),
        output_file=str(output_path),
        models_dir=str(models_dir),
        external_cols=external_cols,
    )

    print("\n" + "=" * 88)
    print("Pipeline completed.")
    print(f"Complete training dataset: {data_path}")
    print(f"Imputed dataset:           {output_path}")
    print(f"Performance summary:       {performance_xlsx}")
    print(f"Saved models:              {models_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
