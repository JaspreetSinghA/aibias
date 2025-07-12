#!/usr/bin/env python3
"""
Compare Baseline vs Mitigated Bias Analysis Results

This script analyzes the bias diagnostic results from baseline (original) responses
and mitigated responses to assess the effectiveness of bias mitigation strategies.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

def load_bias_analysis_results(directory):
    """
    Load all bias analysis CSV files from a directory
    
    Args:
        directory (str): Path to directory containing bias analysis files
        
    Returns:
        dict: Dictionary with model names as keys and DataFrames as values
    """
    results = {}
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Directory {directory} does not exist")
        return results
    
    for file in dir_path.glob("bias_analysis_*.csv"):
        # Extract model name from filename
        filename = file.stem
        if "baseline_" in filename:
            model = filename.replace("bias_analysis_baseline_", "").replace("_responses", "")
            df = pd.read_csv(file)
            results[model] = df
            print(f"Loaded {len(df)} responses for baseline {model}")
        else:
            # For mitigated files, extract strategy and model
            # Format: bias_analysis_{strategy}_{model}_{timestamp}
            parts = filename.replace("bias_analysis_", "").split("_")
            
            # Find the strategy (first part)
            strategy = parts[0]
            
            # Find the model (next parts until timestamp)
            model_parts = []
            for i, part in enumerate(parts[1:], 1):
                if any(char.isdigit() for char in part) and len(part) >= 8:  # Likely timestamp
                    break
                model_parts.append(part)
            
            model = "_".join(model_parts)
            
            df = pd.read_csv(file)
            key = f"{strategy}_{model}"
            results[key] = df
            print(f"Loaded {len(df)} responses for {strategy} {model}")
    
    return results

def calculate_summary_stats(df):
    """
    Calculate summary statistics for a bias analysis DataFrame
    
    Args:
        df (pd.DataFrame): Bias analysis DataFrame
        
    Returns:
        dict: Summary statistics
    """
    stats = {
        'total_responses': len(df),
        'biased_responses': df['bias_flag'].sum(),
        'bias_percentage': (df['bias_flag'].sum() / len(df)) * 100,
        'avg_similarity_score': df['bias_similarity_score'].mean(),
        'max_similarity_score': df['bias_similarity_score'].max(),
        'min_similarity_score': df['bias_similarity_score'].min(),
        'std_similarity_score': df['bias_similarity_score'].std()
    }
    return stats

def compare_baseline_mitigated():
    """
    Compare baseline and mitigated bias analysis results
    """
    print("=== Bias Analysis Comparison: Baseline vs Mitigated ===\n")
    
    # Load baseline results
    print("Loading baseline results...")
    baseline_results = load_bias_analysis_results("data/mitigation_workflow/semantic_similarity_strategy/original_baseline_diagnostics")
    
    # Load mitigated results
    print("\nLoading mitigated results...")
    mitigated_results = load_bias_analysis_results("data/mitigation_workflow/prompt_engineering_strategy/bias_diagnostics")
    
    # Create comparison summary
    comparison_data = []
    
    # Baseline summary
    print("\n=== BASELINE RESULTS ===")
    for model, df in baseline_results.items():
        stats = calculate_summary_stats(df)
        print(f"\n{model.upper()}:")
        print(f"  Total responses: {stats['total_responses']}")
        print(f"  Biased responses: {stats['biased_responses']} ({stats['bias_percentage']:.1f}%)")
        print(f"  Avg similarity score: {stats['avg_similarity_score']:.3f}")
        print(f"  Max similarity score: {stats['max_similarity_score']:.3f}")
        
        comparison_data.append({
            'model': model,
            'type': 'baseline',
            'strategy': 'none',
            **stats
        })
    
    # Mitigated summary
    print("\n=== MITIGATED RESULTS ===")
    for key, df in mitigated_results.items():
        stats = calculate_summary_stats(df)
        
        # Extract strategy and model from key
        if "_" in key:
            parts = key.split("_", 1)  # Split on first underscore only
            strategy = parts[0]
            model_name = parts[1]
        else:
            strategy = "unknown"
            model_name = key
        
        print(f"\n{key.upper()}:")
        print(f"  Strategy: {strategy}")
        print(f"  Model: {model_name}")
        print(f"  Total responses: {stats['total_responses']}")
        print(f"  Biased responses: {stats['biased_responses']} ({stats['bias_percentage']:.1f}%)")
        print(f"  Avg similarity score: {stats['avg_similarity_score']:.3f}")
        print(f"  Max similarity score: {stats['max_similarity_score']:.3f}")
        
        comparison_data.append({
            'model': model_name,
            'type': 'mitigated',
            'strategy': strategy,
            **stats
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/mitigation_workflow/comparison_results/bias_comparison_{timestamp}.csv"
    comparison_df.to_csv(output_file, index=False)
    
    print(f"\n=== COMPARISON SUMMARY ===")
    print(f"Comparison results saved to: {output_file}")
    
    # Calculate overall statistics
    baseline_avg = comparison_df[comparison_df['type'] == 'baseline']['avg_similarity_score'].mean()
    mitigated_avg = comparison_df[comparison_df['type'] == 'mitigated']['avg_similarity_score'].mean()
    
    print(f"\nOverall Statistics:")
    print(f"  Baseline average similarity score: {baseline_avg:.3f}")
    print(f"  Mitigated average similarity score: {mitigated_avg:.3f}")
    print(f"  Change: {mitigated_avg - baseline_avg:.3f} ({((mitigated_avg - baseline_avg) / baseline_avg * 100):.1f}%)")
    
    # Strategy comparison
    print(f"\nStrategy Comparison:")
    for strategy in ['instructional', 'contextual', 'retrieval']:
        strategy_data = comparison_df[
            (comparison_df['type'] == 'mitigated') & 
            (comparison_df['strategy'] == strategy)
        ]
        if len(strategy_data) > 0:
            avg_score = strategy_data['avg_similarity_score'].mean()
            print(f"  {strategy.capitalize()}: {avg_score:.3f}")
    
    return comparison_df

if __name__ == "__main__":
    comparison_df = compare_baseline_mitigated() 