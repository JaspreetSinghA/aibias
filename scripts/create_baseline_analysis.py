#!/usr/bin/env python3
"""
Create Comprehensive Baseline Bias Analysis

This script generates:
1. Summary tables of bias analysis results
2. Bar chart showing % flagged responses by model
3. Histogram showing distribution of bias scores
4. Detailed statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_bias_results(directory):
    """Load all bias analysis results from a directory"""
    results = {}
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Directory {directory} does not exist")
        return results
    
    for file in dir_path.glob("bias_analysis_*.csv"):
        filename = file.stem
        if "baseline_" in filename:
            model = filename.replace("bias_analysis_baseline_", "").replace("_responses", "")
        else:
            # For other files, extract model name
            model = filename.replace("bias_analysis_", "").split("_")[0]
        
        df = pd.read_csv(file)
        results[model] = df
        print(f"Loaded {len(df)} responses for {model}")
    
    return results

def create_summary_table(results, output_dir):
    """Create a comprehensive summary table"""
    summary_data = []
    
    for model, df in results.items():
        stats = {
            'Model': model.upper(),
            'Total Responses': len(df),
            'Flagged Responses': df['bias_flag'].sum(),
            'Flagged Percentage': (df['bias_flag'].sum() / len(df)) * 100,
            'Mean Bias Score': df['bias_similarity_score'].mean(),
            'Median Bias Score': df['bias_similarity_score'].median(),
            'Std Bias Score': df['bias_similarity_score'].std(),
            'Min Bias Score': df['bias_similarity_score'].min(),
            'Max Bias Score': df['bias_similarity_score'].max(),
            'Q1 Bias Score': df['bias_similarity_score'].quantile(0.25),
            'Q3 Bias Score': df['bias_similarity_score'].quantile(0.75)
        }
        summary_data.append(stats)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary table
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"baseline_summary_table_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n=== SUMMARY TABLE ===")
    print(summary_df.to_string(index=False))
    print(f"\nSummary table saved to: {summary_file}")
    
    return summary_df

def create_bar_chart(results, output_dir):
    """Create bar chart showing % flagged responses by model"""
    models = []
    flagged_percentages = []
    
    for model, df in results.items():
        models.append(model.upper())
        flagged_percentages.append((df['bias_flag'].sum() / len(df)) * 100)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, flagged_percentages, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    # Add value labels on bars
    for bar, percentage in zip(bars, flagged_percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Percentage of Flagged Responses by Model', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Percentage Flagged (%)', fontsize=12)
    plt.ylim(0, max(flagged_percentages) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"baseline_flagged_percentage_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Bar chart saved to: {plot_file}")
    return plot_file

def create_histogram(results, output_dir):
    """Create histogram showing distribution of bias scores"""
    plt.figure(figsize=(12, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, (model, df) in enumerate(results.items()):
        plt.hist(df['bias_similarity_score'], bins=20, alpha=0.7, 
                label=model.upper(), color=colors[i], edgecolor='black')
    
    plt.axvline(x=0.7, color='red', linestyle='--', linewidth=2, 
                label='Bias Threshold (0.7)')
    
    plt.title('Distribution of Bias Similarity Scores by Model', fontsize=16, fontweight='bold')
    plt.xlabel('Bias Similarity Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"baseline_score_distribution_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Histogram saved to: {plot_file}")
    return plot_file

def create_boxplot(results, output_dir):
    """Create boxplot showing bias score distributions"""
    plt.figure(figsize=(10, 6))
    
    # Prepare data for boxplot
    data = []
    labels = []
    for model, df in results.items():
        data.append(df['bias_similarity_score'].values)
        labels.append(model.upper())
    
    box_plot = plt.boxplot(data, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.axhline(y=0.7, color='red', linestyle='--', linewidth=2, 
                label='Bias Threshold (0.7)')
    
    plt.title('Bias Score Distribution by Model (Box Plot)', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Bias Similarity Score', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"baseline_boxplot_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Boxplot saved to: {plot_file}")
    return plot_file

def create_detailed_analysis(results, output_dir):
    """Create detailed statistical analysis"""
    analysis_data = []
    
    for model, df in results.items():
        # Basic statistics
        basic_stats = {
            'Model': model.upper(),
            'Total Responses': len(df),
            'Flagged Count': df['bias_flag'].sum(),
            'Flagged Percentage': (df['bias_flag'].sum() / len(df)) * 100
        }
        
        # Score statistics
        scores = df['bias_similarity_score']
        score_stats = {
            'Mean Score': scores.mean(),
            'Median Score': scores.median(),
            'Std Score': scores.std(),
            'Min Score': scores.min(),
            'Max Score': scores.max(),
            'Q1 Score': scores.quantile(0.25),
            'Q3 Score': scores.quantile(0.75),
            'IQR Score': scores.quantile(0.75) - scores.quantile(0.25)
        }
        
        # Threshold analysis
        threshold_analysis = {
            'Above 0.5': (scores > 0.5).sum(),
            'Above 0.6': (scores > 0.6).sum(),
            'Above 0.7': (scores > 0.7).sum(),
            'Above 0.8': (scores > 0.8).sum(),
            'Above 0.9': (scores > 0.9).sum()
        }
        
        # Combine all statistics
        all_stats = {**basic_stats, **score_stats, **threshold_analysis}
        analysis_data.append(all_stats)
    
    analysis_df = pd.DataFrame(analysis_data)
    
    # Save detailed analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = output_dir / f"baseline_detailed_analysis_{timestamp}.csv"
    analysis_df.to_csv(analysis_file, index=False)
    
    print(f"\n=== DETAILED ANALYSIS ===")
    print(analysis_df.to_string(index=False))
    print(f"\nDetailed analysis saved to: {analysis_file}")
    
    return analysis_df

def main():
    """Main function to create comprehensive baseline analysis"""
    print("=== BASELINE BIAS ANALYSIS ===\n")
    
    # Set up directories
    input_dir = "data/mitigation_workflow/semantic_similarity_strategy/original_baseline_diagnostics"
    output_dir = Path("data/mitigation_workflow/semantic_similarity_strategy/baseline_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading bias analysis results...")
    results = load_bias_results(input_dir)
    
    if not results:
        print("No results found. Please run the bias diagnostic tool first.")
        return
    
    print(f"\nLoaded {len(results)} model results")
    
    # Create all analyses
    print("\nCreating summary table...")
    summary_df = create_summary_table(results, output_dir)
    
    print("\nCreating bar chart...")
    bar_chart_file = create_bar_chart(results, output_dir)
    
    print("\nCreating histogram...")
    histogram_file = create_histogram(results, output_dir)
    
    print("\nCreating boxplot...")
    boxplot_file = create_boxplot(results, output_dir)
    
    print("\nCreating detailed analysis...")
    detailed_df = create_detailed_analysis(results, output_dir)
    
    # Create a summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"baseline_analysis_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("BASELINE BIAS ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total models analyzed: {len(results)}\n")
        f.write(f"Total responses analyzed: {sum(len(df) for df in results.values())}\n\n")
        
        f.write("MODEL COMPARISON:\n")
        f.write("-" * 20 + "\n")
        for model, df in results.items():
            flagged_pct = (df['bias_flag'].sum() / len(df)) * 100
            avg_score = df['bias_similarity_score'].mean()
            f.write(f"{model.upper()}: {flagged_pct:.1f}% flagged, avg score: {avg_score:.3f}\n")
        
        f.write(f"\nFiles generated:\n")
        f.write(f"- Summary table: {summary_df.shape[0]} rows\n")
        f.write(f"- Bar chart: {bar_chart_file.name}\n")
        f.write(f"- Histogram: {histogram_file.name}\n")
        f.write(f"- Boxplot: {boxplot_file.name}\n")
        f.write(f"- Detailed analysis: {detailed_df.shape[0]} rows\n")
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"All files saved in: {output_dir}")
    print(f"Report saved as: {report_file}")

if __name__ == "__main__":
    main() 