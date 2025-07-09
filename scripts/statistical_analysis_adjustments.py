#!/usr/bin/env python3
"""
Statistical Analysis of Score Adjustments
Compares original vs adjusted data to identify significant changes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

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

def calculate_statistics(df, label):
    """Calculate comprehensive statistics for a dataset"""
    score_columns = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    stats = {}
    
    for col in score_columns:
        if col in df.columns:
            scores = df[col].dropna()
            if len(scores) > 0:
                stats[col] = {
                    'count': len(scores),
                    'mean': scores.mean(),
                    'median': scores.median(),
                    'std': scores.std(),
                    'min': scores.min(),
                    'max': scores.max(),
                    'q25': scores.quantile(0.25),
                    'q75': scores.quantile(0.75),
                    'skewness': scores.skew(),
                    'kurtosis': scores.kurtosis()
                }
    
    return stats

def compare_distributions(original_stats, adjusted_stats, rater_name):
    """Compare original vs adjusted statistics"""
    print(f"\n{'='*80}")
    print(f"STATISTICAL COMPARISON: {rater_name}")
    print(f"{'='*80}")
    
    score_columns = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    
    for col in score_columns:
        if col in original_stats and col in adjusted_stats:
            orig = original_stats[col]
            adj = adjusted_stats[col]
            
            print(f"\n{col}:")
            print(f"  Mean:     {orig['mean']:.3f} → {adj['mean']:.3f} (Δ: {adj['mean'] - orig['mean']:+.3f})")
            print(f"  Median:   {orig['median']:.3f} → {adj['median']:.3f} (Δ: {adj['median'] - orig['median']:+.3f})")
            print(f"  Std Dev:  {orig['std']:.3f} → {adj['std']:.3f} (Δ: {adj['std'] - orig['std']:+.3f})")
            print(f"  Range:    {orig['min']:.1f}-{orig['max']:.1f} → {adj['min']:.1f}-{adj['max']:.1f}")
            print(f"  Skewness: {orig['skewness']:.3f} → {adj['skewness']:.3f} (Δ: {adj['skewness'] - orig['skewness']:+.3f})")

def analyze_score_distributions(original_df, adjusted_df, rater_name):
    """Analyze score distribution changes"""
    score_columns = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    
    print(f"\n{'='*80}")
    print(f"SCORE DISTRIBUTION ANALYSIS: {rater_name}")
    print(f"{'='*80}")
    
    for col in score_columns:
        if col in original_df.columns and col in adjusted_df.columns:
            orig_scores = original_df[col].dropna()
            adj_scores = adjusted_df[col].dropna()
            
            print(f"\n{col}:")
            
            # Score frequency analysis
            orig_freq = orig_scores.value_counts().sort_index()
            adj_freq = adj_scores.value_counts().sort_index()
            
            print("  Score Frequencies:")
            for score in range(1, 6):
                orig_count = orig_freq.get(score, 0)
                adj_count = adj_freq.get(score, 0)
                change = adj_count - orig_count
                print(f"    {score}: {orig_count} → {adj_count} ({change:+d})")
            
            # High score analysis
            orig_high = (orig_scores >= 4).sum()
            adj_high = (adj_scores >= 4).sum()
            print(f"  Scores ≥ 4: {orig_high} → {adj_high} ({adj_high - orig_high:+d})")
            
            # Perfect score analysis
            orig_perfect = (orig_scores == 5).sum()
            adj_perfect = (adj_scores == 5).sum()
            print(f"  Perfect scores (5): {orig_perfect} → {adj_perfect} ({adj_perfect - orig_perfect:+d})")

def create_comparison_visualizations(original_df, adjusted_df, rater_name):
    """Create visualizations comparing original vs adjusted distributions"""
    
    # Create output directory
    output_dir = "statistical_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    score_columns = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Score Distribution Comparison: {rater_name}\nOriginal vs Adjusted', fontsize=16, fontweight='bold')
    
    for i, col in enumerate(score_columns):
        if col in original_df.columns and col in adjusted_df.columns:
            row = i // 3
            col_idx = i % 3
            ax = axes[row, col_idx]
            
            # Get scores
            orig_scores = original_df[col].dropna()
            adj_scores = adjusted_df[col].dropna()
            
            # Create histogram comparison
            ax.hist(orig_scores, bins=range(1, 7), alpha=0.6, label='Original', color='blue', edgecolor='black')
            ax.hist(adj_scores, bins=range(1, 7), alpha=0.6, label='Adjusted', color='red', edgecolor='black')
            
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            orig_mean = orig_scores.mean()
            adj_mean = adj_scores.mean()
            ax.text(0.02, 0.98, f'Original Mean: {orig_mean:.2f}\nAdjusted Mean: {adj_mean:.2f}\nChange: {adj_mean-orig_mean:+.2f}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Remove the last subplot if not needed
    if len(score_columns) < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{rater_name}_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create box plot comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for box plot
    plot_data = []
    labels = []
    
    for col in score_columns:
        if col in original_df.columns and col in adjusted_df.columns:
            orig_scores = original_df[col].dropna()
            adj_scores = adjusted_df[col].dropna()
            
            plot_data.extend([orig_scores, adj_scores])
            labels.extend([f'{col} (Orig)', f'{col} (Adj)'])
    
    if plot_data:
        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral'] * len(score_columns)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(f'Score Distribution Box Plots: {rater_name}\nOriginal vs Adjusted', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 6)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{rater_name}_boxplot_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

def generate_summary_report(original_stats, adjusted_stats, rater_name):
    """Generate a summary report of key changes"""
    
    score_columns = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    
    print(f"\n{'='*80}")
    print(f"SUMMARY REPORT: {rater_name}")
    print(f"{'='*80}")
    
    # Overall changes
    total_original_mean = np.mean([original_stats[col]['mean'] for col in score_columns if col in original_stats])
    total_adjusted_mean = np.mean([adjusted_stats[col]['mean'] for col in score_columns if col in adjusted_stats])
    
    print(f"Overall Mean Score: {total_original_mean:.3f} → {total_adjusted_mean:.3f} (Δ: {total_adjusted_mean - total_original_mean:+.3f})")
    
    # Most affected categories
    changes = []
    for col in score_columns:
        if col in original_stats and col in adjusted_stats:
            change = adjusted_stats[col]['mean'] - original_stats[col]['mean']
            changes.append((col, change))
    
    changes.sort(key=lambda x: x[1])  # Sort by change magnitude
    
    print(f"\nMost Affected Categories (by mean change):")
    for col, change in changes:
        print(f"  {col}: {change:+.3f}")
    
    # Variance changes
    print(f"\nVariance Changes (std dev):")
    for col in score_columns:
        if col in original_stats and col in adjusted_stats:
            orig_std = original_stats[col]['std']
            adj_std = adjusted_stats[col]['std']
            change = adj_std - orig_std
            print(f"  {col}: {orig_std:.3f} → {adj_std:.3f} ({change:+.3f})")

def main_analysis():
    """Main analysis function"""
    
    print("=" * 80)
    print("STATISTICAL ANALYSIS OF SCORE ADJUSTMENTS")
    print("=" * 80)
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # File mappings
    file_mappings = [
        {
            'rater_file': 'archive/Sikh Biases LLM - Narveer - LLM#2.csv',
            'model_file': 'llm_sikh_bias_responses_Narveer_claude-3-haiku-20240307.csv',
            'rater_name': 'Narveer'
        },
        {
            'rater_file': 'archive/Sikh Biases LLM - Jaspreet - LLM#2.csv',
            'model_file': 'llm_sikh_bias_responses_Jaspreet_claude-3-haiku-20240307.csv',
            'rater_name': 'Jaspreet'
        }
    ]
    
    for mapping in file_mappings:
        print(f"Processing: {mapping['rater_name']}")
        print("-" * 60)
        
        # Load original and adjusted data
        original_df = load_data(mapping['rater_file'])
        adjusted_df = load_data(mapping['model_file'])
        
        if original_df is None or adjusted_df is None:
            print(f"ERROR: Could not load files for {mapping['rater_name']}")
            continue
        
        # Calculate statistics
        original_stats = calculate_statistics(original_df, "Original")
        adjusted_stats = calculate_statistics(adjusted_df, "Adjusted")
        
        # Perform analyses
        compare_distributions(original_stats, adjusted_stats, mapping['rater_name'])
        analyze_score_distributions(original_df, adjusted_df, mapping['rater_name'])
        generate_summary_report(original_stats, adjusted_stats, mapping['rater_name'])
        
        # Create visualizations
        create_comparison_visualizations(original_df, adjusted_df, mapping['rater_name'])
        
        print(f"\nAnalysis completed for {mapping['rater_name']}")
        print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Check 'statistical_analysis_results/' directory for visualizations")

if __name__ == "__main__":
    main_analysis() 