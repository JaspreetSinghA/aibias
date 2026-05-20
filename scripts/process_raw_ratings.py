"""
Process Raw Rater Xlsx Files → Clean Processed CSVs
====================================================
Reads the original Google Sheets exports (xlsx) for the four external raters
(Gurleen, Noor, Anu, Harpreet) and writes one clean CSV per rater to the
processed data directory, replacing the previously duplicated files.

Transformations applied
-----------------------
  1. Extract Prompt ID + 5 score columns only.
  2. Rename "Relevence" → "Relevance" (corrects consistent misspelling in raw sheets).
  3. Convert scores of 0 → NaN: 0 is outside the valid 1–5 scale and indicates a
     blank cell that the spreadsheet defaulted to 0 during export.

Output files (written to DATA_PROCESSED_DIR):
  llm_sikh_bias_responses_Gurleen_gpt-4.csv
  llm_sikh_bias_responses_Noor_gpt-4.csv
  llm_sikh_bias_responses_Anu_llama-3.3-70b-versatile.csv
  llm_sikh_bias_responses_Harpreet_llama-3.3-70b-versatile.csv

Usage
-----
  python3 scripts/process_raw_ratings.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_RAW_DIR = Path(__file__).resolve().parent.parent.parent / "biaslense" / "biaslense" / "data"
DATA_PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

DIMENSIONS = ["Accuracy", "Relevance", "Fairness", "Neutrality", "Representation"]

# Maps output filename → (xlsx path, sheet name)
RATER_SOURCES: dict[str, tuple[Path, str]] = {
    "llm_sikh_bias_responses_Gurleen_gpt-4.csv": (
        DATA_RAW_DIR / "Gurleen(1).xlsx", "LLM#3"
    ),
    "llm_sikh_bias_responses_Noor_gpt-4.csv": (
        DATA_RAW_DIR / "Sikh Biases LLM - Noor.xlsx", "LLM#3"
    ),
    "llm_sikh_bias_responses_Anu_llama-3.3-70b-versatile.csv": (
        DATA_RAW_DIR / "Sikh Biases LLM - Anu Massi.xlsx", "LLM#1"
    ),
    "llm_sikh_bias_responses_Harpreet_llama-3.3-70b-versatile.csv": (
        DATA_RAW_DIR / "Sikh Biases LLM - Harpreet.xlsx", "LLM#1"
    ),
}


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_rater_xlsx(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Read a single rater xlsx sheet and return a clean DataFrame with columns:
    [Prompt ID, Accuracy, Relevance, Fairness, Neutrality, Representation].

    Zeros are replaced with NaN (outside the valid 1–5 scale).
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")

    # Fix misspelling present in all raw sheets
    df = df.rename(columns={"Relevence": "Relevance"})

    for dim in DIMENSIONS:
        df[dim] = pd.to_numeric(df[dim], errors="coerce")
        # 0 is not a valid rating on the 1–5 scale — treat as missing
        df[dim] = df[dim].replace(0, np.nan)

    keep = ["Prompt ID"] + DIMENSIONS
    df = df[keep].copy()
    df = df[df["Prompt ID"].notna()].reset_index(drop=True)

    return df


def validate_output(df: pd.DataFrame, output_name: str) -> None:
    """Print a validation summary for the processed DataFrame."""
    print(f"\n  {output_name}")
    print(f"    Rows: {len(df)}")
    for dim in DIMENSIONS:
        n_valid = df[dim].notna().sum()
        n_missing = df[dim].isna().sum()
        missing_str = f"  ({n_missing} missing)" if n_missing > 0 else ""
        print(f"    {dim}: {n_valid} valid{missing_str}")


def main() -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Processing raw rater xlsx files")
    print(f"Source dir : {DATA_RAW_DIR}")
    print(f"Output dir : {DATA_PROCESSED_DIR}")
    print("=" * 60)

    for out_name, (xlsx_path, sheet) in RATER_SOURCES.items():
        if not xlsx_path.exists():
            print(f"\n  ERROR: source file not found — {xlsx_path}")
            continue

        df = process_rater_xlsx(xlsx_path, sheet)
        out_path = DATA_PROCESSED_DIR / out_name
        df.to_csv(out_path, index=False)
        validate_output(df, out_name)

    print("\n" + "=" * 60)
    print("Done. Processed CSVs written to:")
    print(f"  {DATA_PROCESSED_DIR}")

    # Sanity-check: confirm Gurleen and Noor are no longer identical
    g_path = DATA_PROCESSED_DIR / "llm_sikh_bias_responses_Gurleen_gpt-4.csv"
    n_path = DATA_PROCESSED_DIR / "llm_sikh_bias_responses_Noor_gpt-4.csv"
    if g_path.exists() and n_path.exists():
        g = pd.read_csv(g_path)
        n = pd.read_csv(n_path)
        shared = set(g["Prompt ID"]) & set(n["Prompt ID"])
        diffs = 0
        for pid in shared:
            gr = g[g["Prompt ID"] == pid][DIMENSIONS].values
            nr = n[n["Prompt ID"] == pid][DIMENSIONS].values
            if not np.array_equal(gr, nr):
                diffs += 1
        print(f"\n  GPT-4 pair independence check: {diffs}/{len(shared)} prompts have different scores")
        if diffs == 0:
            print("  WARNING: files are still identical — check source data")
        else:
            print("  OK: rater files are genuinely independent")


if __name__ == "__main__":
    main()
