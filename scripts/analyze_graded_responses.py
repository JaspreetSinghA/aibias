#!/usr/bin/env python3
"""
Comprehensive Analysis of Graded Responses
==========================================

This script analyzes the standardized graded responses to evaluate:
1. Effectiveness of different mitigation strategies
2. Model performance comparisons
3. Category-wise analysis
4. Statistical significance of improvements
5. Qualitative insights from comments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import scipy.stats as stats
from scipy.stats import f_oneway, tukey_hsd

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_standardized_data():
    """Load the standardized graded responses data."""
    data_path = Path("data/mitigation_workflow/prompt_engineering_strategy/standardized_graded_responses/all_standardized_graded_responses.csv")
    
    if not data_path.exists():
        raise FileNotFoundError("Standardized graded responses file not found. Run standardize_graded_responses.py first.")
    
    df = pd.read_csv(data_path)
    
    # Clean the data - remove rows with missing Average_Score
    df_clean = df.dropna(subset=['Average_Score'])
    
    print(f"Original data: {len(df)} rows")
    print(f"After removing missing scores: {len(df_clean)} rows")
    
    return df_clean

def create_strategy_comparison_visualizations(df, output_dir):
    """Create visualizations comparing different mitigation strategies."""
    
    # 1. Strategy effectiveness comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Mitigation Strategy Effectiveness Analysis', fontsize=16, fontweight='bold')
    
    # Average scores by strategy
    strategy_means = df.groupby('Strategy')['Average_Score'].mean().sort_values(ascending=False)
    bars1 = axes[0, 0].bar(strategy_means.index, strategy_means.values, 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    axes[0, 0].set_title('Average Scores by Strategy', fontweight='bold')
    axes[0, 0].set_ylabel('Average Score')
    axes[0, 0].set_ylim(0, 5)
    
    # Add value labels on bars
    for bar, value in zip(bars1, strategy_means.values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Strategy effectiveness by model
    strategy_model_means = df.groupby(['Strategy', 'Model'])['Average_Score'].mean().unstack()
    strategy_model_means.plot(kind='bar', ax=axes[0, 1], alpha=0.8)
    axes[0, 1].set_title('Strategy Effectiveness by Model', fontweight='bold')
    axes[0, 1].set_ylabel('Average Score')
    axes[0, 1].legend(title='Model')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Individual metric scores by strategy
    metrics = ['Accuracy (1-5)', 'Relevance (1-5)', 'Fairness (1-5)', 'Neutrality (1-5)', 'Representation (1-5)']
    strategy_metrics = df.groupby('Strategy')[metrics].mean()
    
    strategy_metrics.plot(kind='bar', ax=axes[1, 0], alpha=0.8)
    axes[1, 0].set_title('Individual Metrics by Strategy', fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Score distribution by strategy
    for strategy in df['Strategy'].unique():
        strategy_data = df[df['Strategy'] == strategy]['Average_Score']
        axes[1, 1].hist(strategy_data, alpha=0.6, label=strategy, bins=10)
    
    axes[1, 1].set_title('Score Distribution by Strategy', fontweight='bold')
    axes[1, 1].set_xlabel('Average Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'strategy_effectiveness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_comparison_visualizations(df, output_dir):
    """Create visualizations comparing different models."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # Average scores by model
    model_means = df.groupby('Model')['Average_Score'].mean().sort_values(ascending=False)
    bars1 = axes[0, 0].bar(model_means.index, model_means.values, 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    axes[0, 0].set_title('Average Scores by Model', fontweight='bold')
    axes[0, 0].set_ylabel('Average Score')
    axes[0, 0].set_ylim(0, 5)
    
    # Add value labels on bars
    for bar, value in zip(bars1, model_means.values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Model performance by category
    model_category_means = df.groupby(['Model', 'Category'])['Average_Score'].mean().unstack()
    model_category_means.plot(kind='bar', ax=axes[0, 1], alpha=0.8)
    axes[0, 1].set_title('Model Performance by Category', fontweight='bold')
    axes[0, 1].set_ylabel('Average Score')
    axes[0, 1].legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Individual metrics by model
    metrics = ['Accuracy (1-5)', 'Relevance (1-5)', 'Fairness (1-5)', 'Neutrality (1-5)', 'Representation (1-5)']
    model_metrics = df.groupby('Model')[metrics].mean()
    
    model_metrics.plot(kind='bar', ax=axes[1, 0], alpha=0.8)
    axes[1, 0].set_title('Individual Metrics by Model', fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Score distribution by model
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]['Average_Score']
        axes[1, 1].hist(model_data, alpha=0.6, label=model, bins=10)
    
    axes[1, 1].set_title('Score Distribution by Model', fontweight='bold')
    axes[1, 1].set_xlabel('Average Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_category_analysis_visualizations(df, output_dir):
    """Create visualizations for category-wise analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Category-wise Analysis', fontsize=16, fontweight='bold')
    
    # Average scores by category
    category_means = df.groupby('Category')['Average_Score'].mean().sort_values(ascending=False)
    bars1 = axes[0, 0].bar(category_means.index, category_means.values, 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    axes[0, 0].set_title('Average Scores by Category', fontweight='bold')
    axes[0, 0].set_ylabel('Average Score')
    axes[0, 0].set_ylim(0, 5)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, category_means.values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Category performance by strategy
    category_strategy_means = df.groupby(['Category', 'Strategy'])['Average_Score'].mean().unstack()
    category_strategy_means.plot(kind='bar', ax=axes[0, 1], alpha=0.8)
    axes[0, 1].set_title('Category Performance by Strategy', fontweight='bold')
    axes[0, 1].set_ylabel('Average Score')
    axes[0, 1].legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Individual metrics by category
    metrics = ['Accuracy (1-5)', 'Relevance (1-5)', 'Fairness (1-5)', 'Neutrality (1-5)', 'Representation (1-5)']
    category_metrics = df.groupby('Category')[metrics].mean()
    
    category_metrics.plot(kind='bar', ax=axes[1, 0], alpha=0.8)
    axes[1, 0].set_title('Individual Metrics by Category', fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Heatmap of category-strategy combinations
    pivot_table = df.pivot_table(values='Average_Score', index='Category', columns='Strategy', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1, 1])
    axes[1, 1].set_title('Category-Strategy Heatmap', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'category_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def perform_statistical_analysis(df):
    """Perform statistical analysis to determine significance of differences."""
    
    # ANOVA test for strategy differences
    strategies = df['Strategy'].unique()
    strategy_groups = [df[df['Strategy'] == strategy]['Average_Score'].dropna() for strategy in strategies]
    
    if len(strategy_groups) >= 2:
        f_stat, p_value = f_oneway(*strategy_groups)
        strategy_anova = {'f_statistic': f_stat, 'p_value': p_value, 'significant': p_value < 0.05}
    else:
        strategy_anova = {'f_statistic': None, 'p_value': None, 'significant': False}
    
    # ANOVA test for model differences
    models = df['Model'].unique()
    model_groups = [df[df['Model'] == model]['Average_Score'].dropna() for model in models]
    
    if len(model_groups) >= 2:
        f_stat, p_value = f_oneway(*model_groups)
        model_anova = {'f_statistic': f_stat, 'p_value': p_value, 'significant': p_value < 0.05}
    else:
        model_anova = {'f_statistic': None, 'p_value': None, 'significant': False}
    
    # ANOVA test for category differences
    categories = df['Category'].unique()
    category_groups = [df[df['Category'] == category]['Average_Score'].dropna() for category in categories]
    
    if len(category_groups) >= 2:
        f_stat, p_value = f_oneway(*category_groups)
        category_anova = {'f_statistic': f_stat, 'p_value': p_value, 'significant': p_value < 0.05}
    else:
        category_anova = {'f_statistic': None, 'p_value': None, 'significant': False}
    
    return {
        'strategy_anova': strategy_anova,
        'model_anova': model_anova,
        'category_anova': category_anova
    }

def generate_comprehensive_report(df, stats_results, output_dir):
    """Generate a comprehensive analysis report."""
    
    report_lines = [
        "COMPREHENSIVE GRADED RESPONSES ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "DATASET OVERVIEW:",
        "-" * 20,
        f"Total responses analyzed: {len(df)}",
        f"Models: {', '.join(df['Model'].unique())}",
        f"Strategies: {', '.join(df['Strategy'].unique())}",
        f"Categories: {', '.join(df['Category'].unique())}",
        f"Overall average score: {df['Average_Score'].mean():.2f}",
        "",
        "STRATEGY EFFECTIVENESS:",
        "-" * 25,
        ""
    ]
    
    # Strategy analysis
    strategy_means = df.groupby('Strategy')['Average_Score'].mean().sort_values(ascending=False)
    for strategy, mean_score in strategy_means.items():
        count = len(df[df['Strategy'] == strategy])
        report_lines.extend([
            f"{strategy.upper()}:",
            f"  Average score: {mean_score:.2f}",
            f"  Number of responses: {count}",
            ""
        ])
    
    # Model analysis
    report_lines.extend([
        "MODEL PERFORMANCE:",
        "-" * 20,
        ""
    ])
    
    model_means = df.groupby('Model')['Average_Score'].mean().sort_values(ascending=False)
    for model, mean_score in model_means.items():
        count = len(df[df['Model'] == model])
        report_lines.extend([
            f"{model}:",
            f"  Average score: {mean_score:.2f}",
            f"  Number of responses: {count}",
            ""
        ])
    
    # Category analysis
    report_lines.extend([
        "CATEGORY PERFORMANCE:",
        "-" * 22,
        ""
    ])
    
    category_means = df.groupby('Category')['Average_Score'].mean().sort_values(ascending=False)
    for category, mean_score in category_means.items():
        count = len(df[df['Category'] == category])
        report_lines.extend([
            f"{category}:",
            f"  Average score: {mean_score:.2f}",
            f"  Number of responses: {count}",
            ""
        ])
    
    # Statistical analysis
    report_lines.extend([
        "STATISTICAL ANALYSIS:",
        "-" * 20,
        ""
    ])
    
    if stats_results['strategy_anova']['significant']:
        report_lines.extend([
            "Strategy Differences: SIGNIFICANT",
            f"  F-statistic: {stats_results['strategy_anova']['f_statistic']:.3f}",
            f"  P-value: {stats_results['strategy_anova']['p_value']:.4f}",
            ""
        ])
    else:
        report_lines.extend([
            "Strategy Differences: NOT SIGNIFICANT",
            f"  P-value: {stats_results['strategy_anova']['p_value']:.4f}",
            ""
        ])
    
    if stats_results['model_anova']['significant']:
        report_lines.extend([
            "Model Differences: SIGNIFICANT",
            f"  F-statistic: {stats_results['model_anova']['f_statistic']:.3f}",
            f"  P-value: {stats_results['model_anova']['p_value']:.4f}",
            ""
        ])
    else:
        report_lines.extend([
            "Model Differences: NOT SIGNIFICANT",
            f"  P-value: {stats_results['model_anova']['p_value']:.4f}",
            ""
        ])
    
    # Key insights
    report_lines.extend([
        "KEY INSIGHTS:",
        "-" * 15,
        "",
        "1. Best Performing Strategy:",
        f"   {strategy_means.index[0].upper()} with average score of {strategy_means.iloc[0]:.2f}",
        "",
        "2. Best Performing Model:",
        f"   {model_means.index[0]} with average score of {model_means.iloc[0]:.2f}",
        "",
        "3. Most Challenging Category:",
        f"   {category_means.index[-1]} with average score of {category_means.iloc[-1]:.2f}",
        "",
        "4. Strategy-Model Combinations:",
        ""
    ])
    
    # Best strategy-model combinations
    combo_means = df.groupby(['Strategy', 'Model'])['Average_Score'].mean().sort_values(ascending=False)
    for i, ((strategy, model), score) in enumerate(combo_means.head(3).items()):
        report_lines.extend([
            f"   {i+1}. {strategy.upper()} + {model}: {score:.2f}",
        ])
    
    report_lines.extend([
        "",
        "FILES GENERATED:",
        "-" * 18,
        "- Strategy effectiveness analysis",
        "- Model performance analysis", 
        "- Category-wise analysis",
        "- Statistical summary tables",
        "- Comprehensive report"
    ])
    
    # Write report
    report_path = output_dir / 'comprehensive_graded_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return report_path

def save_statistical_summaries(df, output_dir):
    """Save detailed statistical summary tables."""
    
    # Strategy summary
    strategy_summary = df.groupby('Strategy').agg({
        'Average_Score': ['count', 'mean', 'std', 'min', 'max'],
        'Accuracy (1-5)': 'mean',
        'Relevance (1-5)': 'mean',
        'Fairness (1-5)': 'mean',
        'Neutrality (1-5)': 'mean',
        'Representation (1-5)': 'mean'
    }).round(3)
    
    strategy_summary.to_csv(output_dir / 'strategy_statistical_summary.csv')
    
    # Model summary
    model_summary = df.groupby('Model').agg({
        'Average_Score': ['count', 'mean', 'std', 'min', 'max'],
        'Accuracy (1-5)': 'mean',
        'Relevance (1-5)': 'mean',
        'Fairness (1-5)': 'mean',
        'Neutrality (1-5)': 'mean',
        'Representation (1-5)': 'mean'
    }).round(3)
    
    model_summary.to_csv(output_dir / 'model_statistical_summary.csv')
    
    # Category summary
    category_summary = df.groupby('Category').agg({
        'Average_Score': ['count', 'mean', 'std', 'min', 'max'],
        'Accuracy (1-5)': 'mean',
        'Relevance (1-5)': 'mean',
        'Fairness (1-5)': 'mean',
        'Neutrality (1-5)': 'mean',
        'Representation (1-5)': 'mean'
    }).round(3)
    
    category_summary.to_csv(output_dir / 'category_statistical_summary.csv')
    
    return [output_dir / 'strategy_statistical_summary.csv',
            output_dir / 'model_statistical_summary.csv',
            output_dir / 'category_statistical_summary.csv']

def main():
    """Main execution function."""
    print("Starting Comprehensive Graded Responses Analysis...")
    
    # Create output directory
    output_dir = Path("data/mitigation_workflow/prompt_engineering_strategy/comprehensive_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading standardized data...")
    df = load_standardized_data()
    
    # Create visualizations
    print("Creating strategy comparison visualizations...")
    create_strategy_comparison_visualizations(df, output_dir)
    
    print("Creating model comparison visualizations...")
    create_model_comparison_visualizations(df, output_dir)
    
    print("Creating category analysis visualizations...")
    create_category_analysis_visualizations(df, output_dir)
    
    # Perform statistical analysis
    print("Performing statistical analysis...")
    stats_results = perform_statistical_analysis(df)
    
    # Generate report
    print("Generating comprehensive report...")
    report_path = generate_comprehensive_report(df, stats_results, output_dir)
    
    # Save statistical summaries
    print("Saving statistical summaries...")
    summary_files = save_statistical_summaries(df, output_dir)
    
    print(f"\nAnalysis complete!")
    print(f"Report saved to: {report_path}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"Statistical summaries saved to: {output_dir}")
    
    # Print key findings
    print(f"\nKey Findings:")
    print(f"- Best strategy: {df.groupby('Strategy')['Average_Score'].mean().idxmax().upper()}")
    print(f"- Best model: {df.groupby('Model')['Average_Score'].mean().idxmax()}")
    print(f"- Overall average score: {df['Average_Score'].mean():.2f}")
    
    return df

if __name__ == "__main__":
    main() 