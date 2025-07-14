#!/usr/bin/env python3
"""
Analysis of Mitigated Strategies
Compares contextual, instructional, and retrieval-based strategies independently.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class MitigatedStrategiesAnalysis:
    def __init__(self, output_dir="data/mitigation_workflow/prompt_engineering_strategy/mitigated_analysis"):
        self.output_dir = Path(output_dir)
        # Always clear the output directory before each run
        if self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = ['Accuracy (1-5)', 'Relevance (1-5)', 'Fairness (1-5)', 
                       'Neutrality (1-5)', 'Representation (1-5)']
        self.strategies = {
            'contextual': 'Contextual Strategy',
            'instructional': 'Instructional Strategy', 
            'retrieval_based': 'Retrieval-Based Strategy'
        }
        
    def load_data(self):
        """Load all mitigated strategy data."""
        data_file = "data/mitigation_workflow/prompt_engineering_strategy/standardized_graded_responses/all_standardized_graded_responses.csv"
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} total responses")
        print(f"Strategies: {df['Strategy'].unique()}")
        print(f"Models: {df['Model'].unique()}")
        print(f"Unique prompts: {df['Prompt ID'].nunique()}")
        return df
    
    def calculate_statistics(self, df):
        """Calculate comprehensive statistics."""
        # Strategy-level statistics
        strategy_stats = df.groupby('Strategy')[self.metrics + ['Average_Score']].agg(['mean', 'std', 'count']).round(3)
        
        # Model-level statistics
        model_stats = df.groupby('Model')[self.metrics + ['Average_Score']].agg(['mean', 'std', 'count']).round(3)
        
        # Strategy-Model interaction
        strategy_model_stats = df.groupby(['Strategy', 'Model'])[self.metrics + ['Average_Score']].mean().round(3)
        
        # Category analysis
        category_stats = df.groupby(['Category', 'Strategy'])[self.metrics + ['Average_Score']].mean().round(3)
        
        return {
            'strategy_stats': strategy_stats,
            'model_stats': model_stats,
            'strategy_model_stats': strategy_model_stats,
            'category_stats': category_stats
        }
    
    def generate_visualizations(self, df, stats):
        """Generate all required visualizations."""
        # 1. Bar Chart: Average Scores by Strategy
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.metrics + ['Average_Score']):
            strategy_means = df.groupby('Strategy')[metric].mean()
            strategy_means.plot(kind='bar', ax=axes[i], color='skyblue', alpha=0.7)
            axes[i].set_title(f'Average {metric} by Strategy')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'average_scores_by_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Line Plot: Score Trends by Strategy
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.metrics + ['Average_Score']):
            strategy_means = df.groupby('Strategy')[metric].mean()
            strategy_means.plot(kind='line', marker='o', ax=axes[i], linewidth=2, markersize=8)
            axes[i].set_title(f'{metric} Trends by Strategy')
            axes[i].set_ylabel('Score')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_trends_by_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Distribution Plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.metrics + ['Average_Score']):
            for strategy in df['Strategy'].unique():
                strategy_data = df[df['Strategy'] == strategy][metric]
                axes[i].hist(strategy_data, alpha=0.6, label=strategy, bins=10)
            axes[i].set_title(f'{metric} Distribution by Strategy')
            axes[i].set_xlabel('Score')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Radar Chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate average scores for each strategy
        strategy_means = df.groupby('Strategy')[self.metrics].mean()
        
        # Number of variables
        N = len(self.metrics)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Plot each strategy
        colors = ['red', 'blue', 'green']
        for i, (strategy, color) in enumerate(zip(strategy_means.index, colors)):
            values = strategy_means.loc[strategy].values.flatten().tolist()
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=strategy, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace(' (1-5)', '') for m in self.metrics])
        ax.set_ylim(0, 5)
        ax.set_title('Strategy Performance Comparison', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'strategy_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Violin Plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.metrics + ['Average_Score']):
            sns.violinplot(data=df, x='Strategy', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric} Distribution by Strategy')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'violin_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Heatmap: Strategy-Model Performance
        pivot_data = df.groupby(['Strategy', 'Model'])['Average_Score'].mean().unstack(fill_value=0)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.2f', cbar_kws={'label': 'Average Score'})
        plt.title('Strategy-Model Performance Heatmap')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'strategy_model_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Category Analysis
        category_pivot = df.groupby(['Category', 'Strategy'])['Average_Score'].mean().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(category_pivot, annot=True, cmap='YlOrRd', fmt='.2f', cbar_kws={'label': 'Average Score'})
        plt.title('Category-Strategy Performance Heatmap')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_strategy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_tables(self, df, stats):
        """Save summary tables."""
        # Strategy statistics
        stats['strategy_stats'].to_csv(self.output_dir / 'strategy_statistics.csv')
        
        # Model statistics  
        stats['model_stats'].to_csv(self.output_dir / 'model_statistics.csv')
        
        # Strategy-Model statistics
        stats['strategy_model_stats'].to_csv(self.output_dir / 'strategy_model_statistics.csv')
        
        # Category statistics
        stats['category_stats'].to_csv(self.output_dir / 'category_statistics.csv')
        
        # Summary table
        summary_data = []
        for strategy in df['Strategy'].unique():
            strategy_data = df[df['Strategy'] == strategy]
            for metric in self.metrics + ['Average_Score']:
                summary_data.append({
                    'Strategy': strategy,
                    'Metric': metric,
                    'Mean': strategy_data[metric].mean(),
                    'Std': strategy_data[metric].std(),
                    'Count': len(strategy_data),
                    'Min': strategy_data[metric].min(),
                    'Max': strategy_data[metric].max()
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'summary_table.csv', index=False)
        
        # Save full dataset
        df.to_csv(self.output_dir / 'full_dataset.csv', index=False)
    
    def generate_report(self, df, stats):
        """Generate markdown report."""
        report = f"""# Mitigated Strategies Analysis Report

## Overview
This report analyzes the performance of three bias mitigation strategies: Contextual, Instructional, and Retrieval-Based.

## Dataset Summary
- **Total Responses**: {len(df)}
- **Strategies**: {', '.join(df['Strategy'].unique())}
- **Models**: {', '.join(df['Model'].unique())}
- **Unique Prompts**: {df['Prompt ID'].nunique()}
- **Categories**: {', '.join(df['Category'].unique())}

## Strategy Performance Summary

### Average Scores by Strategy
"""
        
        # Add strategy performance table
        strategy_means = df.groupby('Strategy')[self.metrics + ['Average_Score']].mean().round(3)
        report += "\n```\n" + strategy_means.to_string() + "\n```\n\n"
        
        # Add model performance
        report += "### Average Scores by Model\n"
        model_means = df.groupby('Model')[self.metrics + ['Average_Score']].mean().round(3)
        report += "```\n" + model_means.to_string() + "\n```\n\n"
        
        # Add category analysis
        report += "### Performance by Category and Strategy\n"
        category_means = df.groupby(['Category', 'Strategy'])['Average_Score'].mean().unstack(fill_value=0).round(3)
        report += "```\n" + category_means.to_string() + "\n```\n\n"
        
        # Add key findings
        best_strategy = df.groupby('Strategy')['Average_Score'].mean().idxmax()
        best_score = df.groupby('Strategy')['Average_Score'].mean().max()
        
        report += f"""## Key Findings

1. **Best Performing Strategy**: {best_strategy} with an average score of {best_score:.3f}
2. **Strategy Comparison**: The three strategies show different strengths across various metrics
3. **Model Performance**: Different models perform differently across strategies
4. **Category Insights**: Performance varies by bias category

## Visualizations

The following visualizations are included:
- Average scores by strategy
- Score trends and distributions  
- Radar chart comparing all strategies
- Violin plots showing score distributions
- Heatmaps for strategy-model and category-strategy performance

## Recommendations

Based on the analysis:
1. Consider the specific bias category when choosing a mitigation strategy
2. Different models may benefit from different strategies
3. Further analysis of prompt-specific performance may reveal additional insights

---
*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(self.output_dir / 'analysis_report.md', 'w') as f:
            f.write(report)
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("Loading data...")
        df = self.load_data()
        
        print("Calculating statistics...")
        stats = self.calculate_statistics(df)
        
        print("Generating visualizations...")
        self.generate_visualizations(df, stats)
        
        print("Saving tables...")
        self.save_tables(df, stats)
        
        print("Generating report...")
        self.generate_report(df, stats)
        
        print(f"Analysis complete! Results saved to: {self.output_dir}")

if __name__ == "__main__":
    analyzer = MitigatedStrategiesAnalysis()
    analyzer.run_analysis() 