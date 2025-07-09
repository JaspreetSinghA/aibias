#!/usr/bin/env python3
"""
Count Perfect Scores (5s) for Each Model
Analyzes the distribution of perfect scores across models and raters
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path):
    """Load CSV and handle missing values"""
    try:
        df = pd.read_csv(file_path)
        score_columns = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
        for col in score_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def analyze_perfect_scores(df, model_name, rater_name):
    """Analyze perfect scores (5s) in a dataset"""
    score_columns = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    
    print(f"\n{'='*80}")
    print(f"PERFECT SCORES ANALYSIS: {rater_name} → {model_name}")
    print(f"{'='*80}")
    
    total_scores = 0
    total_perfect = 0
    category_perfect = {}
    rows_with_perfect = 0
    rows_all_perfect = 0
    
    # Analyze each category
    for col in score_columns:
        if col in df.columns:
            scores = df[col].dropna()
            total_scores += len(scores)
            perfect_count = (scores == 5).sum()
            total_perfect += perfect_count
            category_perfect[col] = perfect_count
            
            print(f"{col}: {perfect_count} perfect scores out of {len(scores)} total ({perfect_count/len(scores)*100:.1f}%)")
    
    # Analyze rows with perfect scores
    perfect_scores_per_row = []
    for _, row in df.iterrows():
        row_scores = []
        for col in score_columns:
            if col in df.columns and not pd.isna(row[col]):
                row_scores.append(row[col])
        
        if row_scores:
            perfect_in_row = sum(1 for score in row_scores if score == 5)
            perfect_scores_per_row.append(perfect_in_row)
            
            if perfect_in_row > 0:
                rows_with_perfect += 1
            if perfect_in_row == len(row_scores):  # All scores in row are 5
                rows_all_perfect += 1
    
    print(f"\nOverall Statistics:")
    print(f"Total scores: {total_scores}")
    print(f"Total perfect scores (5s): {total_perfect}")
    print(f"Percentage of perfect scores: {total_perfect/total_scores*100:.1f}%")
    print(f"Rows with at least one perfect score: {rows_with_perfect}")
    print(f"Rows with ALL perfect scores: {rows_all_perfect}")
    
    if perfect_scores_per_row:
        print(f"Average perfect scores per row: {np.mean(perfect_scores_per_row):.2f}")
        print(f"Max perfect scores in a single row: {max(perfect_scores_per_row)}")
    
    return {
        'total_scores': total_scores,
        'total_perfect': total_perfect,
        'percentage_perfect': total_perfect/total_scores*100 if total_scores > 0 else 0,
        'category_perfect': category_perfect,
        'rows_with_perfect': rows_with_perfect,
        'rows_all_perfect': rows_all_perfect,
        'avg_perfect_per_row': np.mean(perfect_scores_per_row) if perfect_scores_per_row else 0
    }

def main_analysis():
    """Main analysis function"""
    
    print("=" * 80)
    print("PERFECT SCORES (5s) ANALYSIS")
    print("=" * 80)
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # File mappings
    file_mappings = [
        {
            'rater_file': 'archive/Sikh Biases LLM - Narveer - LLM#2.csv',
            'model_file': 'llm_sikh_bias_responses_Narveer_claude-3-haiku-20240307.csv',
            'rater_name': 'Narveer',
            'model_name': 'claude-3-haiku-20240307'
        },
        {
            'rater_file': 'archive/Sikh Biases LLM - Jaspreet - LLM#2.csv',
            'model_file': 'llm_sikh_bias_responses_Jaspreet_claude-3-haiku-20240307.csv',
            'rater_name': 'Jaspreet',
            'model_name': 'claude-3-haiku-20240307'
        }
    ]
    
    all_results = {}
    
    for mapping in file_mappings:
        print(f"Processing: {mapping['rater_name']} → {mapping['model_name']}")
        print("-" * 60)
        
        # Load original and adjusted data
        original_df = load_data(mapping['rater_file'])
        adjusted_df = load_data(mapping['model_file'])
        
        if original_df is None or adjusted_df is None:
            print(f"ERROR: Could not load files for {mapping['rater_name']}")
            continue
        
        # Analyze original data
        print("\nORIGINAL DATA (Rater Scores):")
        original_results = analyze_perfect_scores(original_df, "Original", mapping['rater_name'])
        
        # Analyze adjusted data
        print("\nADJUSTED DATA (Model Scores):")
        adjusted_results = analyze_perfect_scores(adjusted_df, mapping['model_name'], mapping['rater_name'])
        
        # Store results
        all_results[mapping['rater_name']] = {
            'original': original_results,
            'adjusted': adjusted_results
        }
        
        print(f"\nAnalysis completed for {mapping['rater_name']}")
        print()
    
    # Summary comparison
    print("=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    for rater_name, results in all_results.items():
        print(f"\n{rater_name}:")
        orig = results['original']
        adj = results['adjusted']
        
        print(f"  Original perfect scores: {orig['total_perfect']} ({orig['percentage_perfect']:.1f}%)")
        print(f"  Adjusted perfect scores: {adj['total_perfect']} ({adj['percentage_perfect']:.1f}%)")
        print(f"  Change: {adj['total_perfect'] - orig['total_perfect']:+d} ({adj['percentage_perfect'] - orig['percentage_perfect']:+.1f}%)")
        
        print(f"  Rows with perfect scores:")
        print(f"    Original: {orig['rows_with_perfect']}")
        print(f"    Adjusted: {adj['rows_with_perfect']}")
        print(f"    Change: {adj['rows_with_perfect'] - orig['rows_with_perfect']:+d}")
        
        print(f"  Rows with ALL perfect scores:")
        print(f"    Original: {orig['rows_all_perfect']}")
        print(f"    Adjusted: {adj['rows_all_perfect']}")
        print(f"    Change: {adj['rows_all_perfect'] - orig['rows_all_perfect']:+d}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main_analysis() 