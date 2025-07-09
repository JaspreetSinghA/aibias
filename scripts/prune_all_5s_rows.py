#!/usr/bin/env python3
"""
Prune All-5s Rows from Model Files
Removes any row where all five scores are 5.
"""

import pandas as pd
from datetime import datetime

def prune_all_5s_rows(file_path):
    df = pd.read_csv(file_path)
    score_columns = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    before = len(df)
    # Only keep rows where not all scores are 5
    mask = ~df[score_columns].apply(lambda row: all(row == 5), axis=1)
    pruned_df = df[mask].reset_index(drop=True)
    after = len(pruned_df)
    pruned_count = before - after
    pruned_df.to_csv(file_path, index=False)
    print(f"{file_path}: Pruned {pruned_count} all-5s rows (kept {after} rows)")
    return pruned_count, after

def main():
    print("=" * 80)
    print("PRUNING ALL-5s ROWS FROM MODEL FILES")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    files = [
        'llm_sikh_bias_responses_Narveer_claude-3-haiku-20240307.csv',
        'llm_sikh_bias_responses_Jaspreet_claude-3-haiku-20240307.csv'
    ]
    total_pruned = 0
    for file_path in files:
        pruned, kept = prune_all_5s_rows(file_path)
        total_pruned += pruned
    print(f"\nTotal all-5s rows pruned: {total_pruned}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main() 