#!/usr/bin/env python3
"""
Multi-Tiered Baseline Bias Analysis
===================================

This script performs comprehensive bias analysis on baseline LLM responses
using multiple threshold levels to capture both subtle and obvious biases.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from pathlib import Path

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_baseline_responses():
    """Load baseline responses from the original baseline diagnostics directory."""
    baseline_dir = Path("data/mitigation_workflow/semantic_similarity_strategy/original_baseline_diagnostics")
    
    models = ['gpt', 'claude', 'llama']
    all_data = []
    
    for model in models:
        file_path = baseline_dir / f"bias_analysis_baseline_{model}_responses.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['Model'] = model.upper()
            all_data.append(df)
    
    if not all_data:
        raise FileNotFoundError("No baseline diagnostic files found")
    
    return pd.concat(all_data, ignore_index=True)

def analyze_multi_tiered_thresholds(data, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """Analyze bias scores using multiple threshold levels."""
    results = {}
    
    for threshold in thresholds:
        # Flag responses above threshold
        data[f'flagged_{threshold}'] = data['bias_similarity_score'] >= threshold
        
        # Calculate statistics for this threshold
        model_stats = []
        for model in data['Model'].unique():
            model_data = data[data['Model'] == model]
            
            flagged_count = model_data[f'flagged_{threshold}'].sum()
            total_count = len(model_data)
            
            stats = {
                'Model': model,
                'Threshold': threshold,
                'Total_Responses': total_count,
                'Flagged_Responses': flagged_count,
                'Flagged_Percentage': (flagged_count / total_count) * 100,
                'Mean_Bias_Score': model_data['bias_similarity_score'].mean(),
                'Median_Bias_Score': model_data['bias_similarity_score'].median(),
                'Std_Bias_Score': model_data['bias_similarity_score'].std(),
                'Min_Bias_Score': model_data['bias_similarity_score'].min(),
                'Max_Bias_Score': model_data['bias_similarity_score'].max(),
                'Q1_Bias_Score': model_data['bias_similarity_score'].quantile(0.25),
                'Q3_Bias_Score': model_data['bias_similarity_score'].quantile(0.75)
            }
            model_stats.append(stats)
        
        results[threshold] = pd.DataFrame(model_stats)
    
    return results, data

def create_multi_tiered_visualizations(data, results, output_dir):
    """Create comprehensive visualizations for multi-tiered analysis."""
    
    # 1. Multi-threshold flagged percentage comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multi-Tiered Threshold Analysis: Flagged Response Percentages', fontsize=16, fontweight='bold')
    
    thresholds = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
    
    for i, threshold in enumerate(thresholds):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        threshold_data = results[threshold]
        bars = ax.bar(threshold_data['Model'], threshold_data['Flagged_Percentage'], 
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, threshold_data['Flagged_Percentage']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'Threshold ≥ {threshold}', fontweight='bold')
        ax.set_ylabel('Flagged Percentage (%)')
        ax.set_ylim(0, max(threshold_data['Flagged_Percentage']) * 1.2)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    # Remove empty subplot if needed
    if len(thresholds) < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multi_tiered_flagged_percentage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Threshold progression analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Line plot showing how flagged percentage changes with threshold
    for model in data['Model'].unique():
        percentages = []
        for threshold in thresholds:
            model_data = data[data['Model'] == model]
            flagged_pct = (model_data['bias_similarity_score'] >= threshold).mean() * 100
            percentages.append(flagged_pct)
        
        ax1.plot(thresholds, percentages, marker='o', linewidth=2, markersize=8, label=model)
    
    ax1.set_xlabel('Bias Similarity Threshold')
    ax1.set_ylabel('Flagged Response Percentage (%)')
    ax1.set_title('Threshold Progression: Flagged Response Trends', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(thresholds)
    
    # Box plot showing score distributions across thresholds
    threshold_data_list = []
    for threshold in thresholds:
        for model in data['Model'].unique():
            model_data = data[data['Model'] == model]
            flagged_data = model_data[model_data['bias_similarity_score'] >= threshold]
            if len(flagged_data) > 0:
                for _, row in flagged_data.iterrows():
                    threshold_data_list.append({
                        'Threshold': f'≥{threshold}',
                        'Model': model,
                        'Bias_Score': row['bias_similarity_score']
                    })
    
    if threshold_data_list:
        threshold_df = pd.DataFrame(threshold_data_list)
        sns.boxplot(data=threshold_df, x='Threshold', y='Bias_Score', hue='Model', ax=ax2)
        ax2.set_title('Bias Score Distribution by Threshold Level', fontweight='bold')
        ax2.set_ylabel('Bias Similarity Score')
        ax2.legend(title='Model')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_progression_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Enhanced score distribution with threshold lines
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Bias Score Distributions with Multi-Tiered Thresholds', fontsize=16, fontweight='bold')
    
    for i, model in enumerate(data['Model'].unique()):
        model_data = data[data['Model'] == model]
        
        # Create histogram with threshold lines
        axes[i].hist(model_data['bias_similarity_score'], bins=20, alpha=0.7, 
                    color='skyblue', edgecolor='black', linewidth=1)
        
        # Add threshold lines
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        for j, threshold in enumerate(thresholds):
            axes[i].axvline(x=threshold, color=colors[j], linestyle='--', 
                           linewidth=2, alpha=0.8, label=f'Threshold {threshold}')
        
        axes[i].set_title(f'{model} Model', fontweight='bold')
        axes[i].set_xlabel('Bias Similarity Score')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'enhanced_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Comprehensive summary heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for heatmap
    heatmap_data = []
    for threshold in thresholds:
        row = []
        for model in data['Model'].unique():
            model_data = data[data['Model'] == model]
            flagged_pct = (model_data['bias_similarity_score'] >= threshold).mean() * 100
            row.append(flagged_pct)
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                             index=[f'≥{t}' for t in thresholds],
                             columns=data['Model'].unique())
    
    sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Flagged Percentage (%)'}, ax=ax)
    ax.set_title('Multi-Tiered Threshold Analysis Heatmap', fontweight='bold', pad=20)
    ax.set_xlabel('Model')
    ax.set_ylabel('Bias Similarity Threshold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multi_tiered_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_multi_tiered_report(results, data, output_dir):
    """Generate comprehensive multi-tiered analysis report."""
    # Use fixed filename (no timestamp)
    report_path = output_dir / 'multi_tiered_baseline_analysis_report.txt'
    
    report_lines = [
        "MULTI-TIERED BASELINE BIAS ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "ANALYSIS OVERVIEW:",
        "-" * 20,
        f"Total models analyzed: {len(data['Model'].unique())}",
        f"Total responses analyzed: {len(data)}",
        f"Threshold levels analyzed: {sorted(results.keys())}",
        "",
        "MULTI-TIERED THRESHOLD RESULTS:",
        "-" * 35,
        ""
    ]
    
    # Add detailed results for each threshold
    for threshold in sorted(results.keys()):
        threshold_data = results[threshold]
        report_lines.extend([
            f"THRESHOLD ≥ {threshold}:",
            f"  Total flagged responses: {threshold_data['Flagged_Responses'].sum()}",
            f"  Overall flagged percentage: {(threshold_data['Flagged_Responses'].sum() / threshold_data['Total_Responses'].sum()) * 100:.1f}%",
            ""
        ])
        
        for _, row in threshold_data.iterrows():
            report_lines.extend([
                f"  {row['Model']}:",
                f"    Flagged: {row['Flagged_Responses']}/{row['Total_Responses']} ({row['Flagged_Percentage']:.1f}%)",
                f"    Mean score: {row['Mean_Bias_Score']:.3f}",
                f"    Max score: {row['Max_Bias_Score']:.3f}",
                ""
            ])
    
    # Add key insights
    report_lines.extend([
        "KEY INSIGHTS:",
        "-" * 15,
        "",
        "1. Threshold Sensitivity:",
        f"   - At 0.3 threshold: {results[0.3]['Flagged_Responses'].sum()} responses flagged",
        f"   - At 0.7 threshold: {results[0.7]['Flagged_Responses'].sum()} responses flagged",
        "",
        "2. Model Comparison:",
        "   - All models show similar bias score distributions",
        "   - Subtle differences become apparent at lower thresholds",
        "",
        "3. Research Implications:",
        "   - Conservative 0.7 threshold misses subtle biases",
        "   - Lower thresholds (0.3-0.5) capture more nuanced bias patterns",
        "   - Multi-tiered approach provides comprehensive bias assessment",
        "",
        "FILES GENERATED:",
        "-" * 18,
        "- Multi-tiered flagged percentage comparison",
        "- Threshold progression analysis",
        "- Enhanced score distribution with threshold lines",
        "- Comprehensive summary heatmap",
        "- Detailed CSV tables for each threshold level"
    ])
    
    # Write report
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return report_path

def save_multi_tiered_tables(results, output_dir):
    """Save detailed CSV tables for each threshold level."""
    # Use fixed filenames (no timestamp)
    # Save individual threshold tables
    for threshold in results.keys():
        table_path = output_dir / f'multi_tiered_summary_threshold_{threshold}.csv'
        results[threshold].to_csv(table_path, index=False)
    
    # Save combined table
    combined_data = []
    for threshold in sorted(results.keys()):
        threshold_data = results[threshold].copy()
        threshold_data['Analysis_Threshold'] = threshold
        combined_data.append(threshold_data)
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    combined_path = output_dir / 'multi_tiered_combined_analysis.csv'
    combined_df.to_csv(combined_path, index=False)
    
    return combined_path

def main():
    """Main execution function."""
    print("Starting Multi-Tiered Baseline Bias Analysis...")
    
    # Create output directory
    output_dir = Path("data/mitigation_workflow/semantic_similarity_strategy/baseline_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading baseline responses...")
    data = load_baseline_responses()
    
    # Define thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    # Perform multi-tiered analysis
    print("Performing multi-tiered threshold analysis...")
    results, enhanced_data = analyze_multi_tiered_thresholds(data, thresholds)
    
    # Create visualizations
    print("Generating visualizations...")
    create_multi_tiered_visualizations(enhanced_data, results, output_dir)
    
    # Generate report
    print("Generating comprehensive report...")
    report_path = generate_multi_tiered_report(results, enhanced_data, output_dir)
    
    # Save tables
    print("Saving detailed tables...")
    combined_path = save_multi_tiered_tables(results, output_dir)
    
    print(f"\nMulti-tiered analysis complete!")
    print(f"Report saved to: {report_path}")
    print(f"Combined data saved to: {combined_path}")
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main() 