#!/usr/bin/env python3
"""
Check data overlap and scan for zero values in metrics. Clean zeros in baseline data.
"""

import pandas as pd
from pathlib import Path
import numpy as np

def clean_zeros_in_baseline():
    print("=== CLEANING ZEROS IN BASELINE DATA ===\n")
    baseline_files = [
        'data/llm_sikh_bias_responses_Narveer_claude-3-haiku-20240307_adjusted_20250708_175346.csv',
        'data/llm_sikh_bias_responses_Noor_gpt-4.csv',
        'data/llm_sikh_bias_responses_Jaspreet_claude-3-haiku-20240307.csv',
        'data/llm_sikh_bias_responses_Harpreet_llama-3.3-70b-versatile.csv',
        'data/llm_sikh_bias_responses_Gurleen_gpt-4.csv',
        'data/llm_sikh_bias_responses_Anu_llama-3.3-70b-versatile.csv'
    ]
    metrics = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    for file in baseline_files:
        if Path(file).exists():
            df = pd.read_csv(file)
            changed = False
            for metric in metrics:
                if metric in df.columns:
                    zero_mask = df[metric] == 0
                    if zero_mask.any():
                        print(f"  Cleaning {zero_mask.sum()} zero(s) in {metric} in {file}")
                        df.loc[zero_mask, metric] = np.nan
                        changed = True
            if changed:
                df.to_csv(file, index=False)
                print(f"  Saved cleaned file: {file}")
    print("\n=== CLEANING COMPLETE ===\n")

def check_data_overlap():
    print("=== CHECKING DATA OVERLAP BETWEEN BASELINE AND MITIGATED DATASETS ===\n")
    
    # Load baseline data
    print("Loading baseline data...")
    baseline_files = [
        'data/llm_sikh_bias_responses_Narveer_claude-3-haiku-20240307_adjusted_20250708_175346.csv',
        'data/llm_sikh_bias_responses_Noor_gpt-4.csv',
        'data/llm_sikh_bias_responses_Jaspreet_claude-3-haiku-20240307.csv',
        'data/llm_sikh_bias_responses_Harpreet_llama-3.3-70b-versatile.csv',
        'data/llm_sikh_bias_responses_Gurleen_gpt-4.csv',
        'data/llm_sikh_bias_responses_Anu_llama-3.3-70b-versatile.csv'
    ]
    
    baseline_data = []
    for file in baseline_files:
        if Path(file).exists():
            df = pd.read_csv(file)
            baseline_data.append(df)
            print(f"  Loaded {file}: {len(df)} rows")
    
    if baseline_data:
        baseline_df = pd.concat(baseline_data, ignore_index=True)
        print(f"Total baseline rows: {len(baseline_df)}")
    else:
        print("No baseline files found!")
        return
    
    # Load mitigated data
    print("\nLoading mitigated data...")
    mitigated_file = "data/mitigation_workflow/prompt_engineering_strategy/standardized_graded_responses/all_standardized_graded_responses.csv"
    
    if Path(mitigated_file).exists():
        mitigated_df = pd.read_csv(mitigated_file)
        print(f"Total mitigated rows: {len(mitigated_df)}")
    else:
        print(f"Mitigated file not found: {mitigated_file}")
        return
    
    # Scan for zeros in metric columns
    print("\n=== SCANNING FOR ZERO VALUES IN METRICS ===")
    baseline_metrics = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    mitigated_metrics = ['Accuracy (1-5)', 'Relevance (1-5)', 'Fairness (1-5)', 'Neutrality (1-5)', 'Representation (1-5)']
    
    print("\nBaseline data:")
    for metric in baseline_metrics:
        if metric in baseline_df.columns:
            zeros = baseline_df[baseline_df[metric] == 0]
            if not zeros.empty:
                print(f"  Found {len(zeros)} zero(s) in {metric}:")
                print(zeros[['Prompt ID', 'Model', metric]])
    
    print("\nMitigated data:")
    for metric in mitigated_metrics:
        if metric in mitigated_df.columns:
            zeros = mitigated_df[mitigated_df[metric] == 0]
            if not zeros.empty:
                print(f"  Found {len(zeros)} zero(s) in {metric}:")
                print(zeros[['Prompt ID', 'Model', metric]])
    
    print("\n=== SCAN COMPLETE ===")

if __name__ == "__main__":
    check_data_overlap() 