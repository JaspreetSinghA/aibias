#!/usr/bin/env python3
"""
Comprehensive Sikh Bias Mitigation Analysis
Merges baseline and mitigated data, calculates improvements, and generates all visualizations and reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import shutil
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ComprehensiveMitigationAnalysis:
    def __init__(self, output_dir="data/mitigation_workflow/prompt_engineering_strategy/comprehensive_analysis"):
        self.output_dir = Path(output_dir)
        # Always clear the output directory before each run
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Strategy labels
        self.strategies = {
            'contextual': 'Contextual Strategy',
            'instructional': 'Instructional Strategy', 
            'retrieval_based': 'Retrieval-Based Strategy'
        }
        
        # Metrics for analysis (baseline uses old names, mitigated uses new names)
        self.metrics = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
        self.metrics_mitigated = ['Accuracy (1-5)', 'Relevance (1-5)', 'Fairness (1-5)', 'Neutrality (1-5)', 'Representation (1-5)']
        
    def load_baseline_data(self):
        """Load all baseline files and combine them."""
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
                df['Strategy'] = 'Baseline'
                baseline_data.append(df)
        
        if baseline_data:
            baseline_df = pd.concat(baseline_data, ignore_index=True)
            print(f"Loaded baseline data: {len(baseline_df)} rows")
            return baseline_df
        else:
            raise FileNotFoundError("No baseline files found")
    
    def load_mitigated_data(self):
        """Load all mitigated strategy files."""
        # Use the consolidated standardized file
        strategy_dir = "data/mitigation_workflow/prompt_engineering_strategy/standardized_graded_responses"
        consolidated_file = Path(strategy_dir) / "all_standardized_graded_responses.csv"
        
        if consolidated_file.exists():
            df = pd.read_csv(consolidated_file)
            print(f"Loaded mitigated data: {len(df)} rows")
            return df
        else:
            raise FileNotFoundError(f"Consolidated file not found: {consolidated_file}")
    
    def merge_baseline_mitigated(self, baseline_df, mitigated_df):
        """Merge baseline and mitigated data by Prompt ID and Model."""
        
        # Create model name mapping
        model_mapping = {
            'CLAUDE': 'claude-3-haiku-20240307',
            'GPT': 'gpt-4', 
            'LLAMA': 'llama-3.3-70b-versatile'
        }
        
        # Standardize model names in mitigated data
        mitigated_df['Model_Original'] = mitigated_df['Model']
        mitigated_df['Model'] = mitigated_df['Model'].map(model_mapping)
        
        # Standardize metric column names in mitigated data
        metric_rename = {
            'Accuracy (1-5)': 'Accuracy',
            'Relevance (1-5)': 'Relevance',
            'Fairness (1-5)': 'Fairness',
            'Neutrality (1-5)': 'Neutrality',
            'Representation (1-5)': 'Representation'
        }
        mitigated_df = mitigated_df.rename(columns=metric_rename)
        
        # Ensure we only include prompts present in all strategies
        baseline_prompts = set(zip(baseline_df['Prompt ID'], baseline_df['Model']))
        mitigated_prompts = set(zip(mitigated_df['Prompt ID'], mitigated_df['Model']))
        
        common_prompts = baseline_prompts.intersection(mitigated_prompts)
        print(f"Common prompts across all strategies: {len(common_prompts)}")
        
        # Filter data to only include common prompts
        baseline_filtered = baseline_df[
            baseline_df.apply(lambda x: (x['Prompt ID'], x['Model']) in common_prompts, axis=1)
        ]
        mitigated_filtered = mitigated_df[
            mitigated_df.apply(lambda x: (x['Prompt ID'], x['Model']) in common_prompts, axis=1)
        ]
        
        # Combine baseline and mitigated data
        combined_df = pd.concat([baseline_filtered, mitigated_filtered], ignore_index=True)
        
        # Calculate bias composite score for all rows
        combined_df['Bias_Composite'] = combined_df[['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']].mean(axis=1)
        
        return combined_df
    
    def calculate_improvements(self, combined_df):
        """Calculate Δ% improvements for each strategy vs baseline."""
        improvements = []
        
        for (prompt_id, model) in combined_df.groupby(['Prompt ID', 'Model']).groups.keys():
            baseline_data = combined_df[
                (combined_df['Prompt ID'] == prompt_id) & 
                (combined_df['Model'] == model) & 
                (combined_df['Strategy'] == 'Baseline')
            ]
            
            if len(baseline_data) == 0:
                continue
                
            baseline_scores = baseline_data.iloc[0][self.metrics + ['Bias_Composite']]
            
            for strategy in self.strategies.keys():
                strategy_data = combined_df[
                    (combined_df['Prompt ID'] == prompt_id) & 
                    (combined_df['Model'] == model) & 
                    (combined_df['Strategy'] == strategy)
                ]
                
                if len(strategy_data) == 0:
                    continue
                    
                # Use mitigated metrics for strategy data
                strategy_scores = strategy_data.iloc[0][self.metrics + ['Bias_Composite']]
                
                # Calculate Δ% improvements
                for i, metric in enumerate(self.metrics + ['Bias_Composite']):
                    baseline_score = baseline_scores[metric]
                    if metric == 'Bias_Composite':
                        strategy_score = strategy_scores[metric]
                    else:
                        strategy_score = strategy_scores[self.metrics[i]]
                    
                    if baseline_score == 0:
                        improvement = np.nan
                    else:
                        improvement = ((strategy_score - baseline_score) / baseline_score) * 100
                    
                    improvements.append({
                        'Prompt ID': prompt_id,
                        'Model': model,
                        'Strategy': strategy,
                        'Metric': metric,
                        'Baseline_Score': baseline_score,
                        'Strategy_Score': strategy_score,
                        'Improvement_Percent': improvement
                    })
        
        improvements_df = pd.DataFrame(improvements)
        return improvements_df
    
    def create_summary_statistics(self, combined_df, improvements_df):
        """Create comprehensive summary statistics."""
        # Overall statistics by strategy
        strategy_stats = combined_df.groupby('Strategy')[self.metrics + ['Bias_Composite']].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(3)
        
        # Improvement statistics
        improvement_stats = improvements_df.groupby(['Strategy', 'Metric'])['Improvement_Percent'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(3)
        
        # Model-wise statistics
        model_stats = combined_df.groupby(['Strategy', 'Model'])['Bias_Composite'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(3)
        
        return strategy_stats, improvement_stats, model_stats
    
    def generate_visualizations(self, combined_df, improvements_df):
        """Generate all required visualizations."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from pathlib import Path
        from matplotlib.ticker import MaxNLocator
        
        # Consistent colorblind-friendly palette
        palette = sns.color_palette("colorblind")
        strategy_order = ['Baseline', 'contextual', 'instructional', 'retrieval_based']
        metric_order = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Composite']
        
        # 1. Bar Chart: Average Scores by Strategy (with standard error error bars, lighter and thinner)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        for i, metric in enumerate(metric_order):
            means = combined_df.groupby('Strategy')[metric].mean().reindex(strategy_order)
            counts = combined_df.groupby('Strategy')[metric].count().reindex(strategy_order)
            stds = combined_df.groupby('Strategy')[metric].std().reindex(strategy_order)
            stderrs = stds / counts.pow(0.5)
            bars = axes[i].bar(means.index, means.values, yerr=stderrs.values, color=palette[:len(strategy_order)], alpha=0.7, capsize=3, ecolor='gray', error_kw={'elinewidth':1, 'capthick':1, 'alpha':0.7})
            axes[i].set_title(f'Average {metric} by Strategy', fontsize=18)
            axes[i].set_ylabel(metric, fontsize=14)
            axes[i].set_xlabel('Strategy', fontsize=14)
            axes[i].tick_params(axis='x', rotation=20, labelsize=12)
            axes[i].grid(True, axis='y', linestyle='--', alpha=0.5)
            axes[i].set_ylim(1, 5)
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[i].annotate(f'{height:.2f}',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 5),
                                 textcoords="offset points",
                                 ha='center', va='bottom', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'average_scores_by_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()
        # Figure caption: Bar chart with standard error error bars (thin, light gray) for each metric by strategy. Error bars represent standard error of the mean.
        
        # 2. Violin Plots: Overlay individual data points, mark outliers, y-axis 0-5
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        for i, metric in enumerate(metric_order):
            sns.violinplot(data=combined_df, x='Strategy', y=metric, order=strategy_order, ax=axes[i], palette=palette, inner=None, cut=0)
            sns.stripplot(data=combined_df, x='Strategy', y=metric, order=strategy_order, ax=axes[i], color='k', size=4, jitter=True, alpha=0.7)
            # Mark outliers (1.5*IQR rule)
            for j, strategy in enumerate(strategy_order):
                vals = combined_df[combined_df['Strategy'] == strategy][metric].dropna()
                if len(vals) > 0:
                    q1, q3 = np.percentile(vals, [25, 75])
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
                    outliers = vals[(vals < lower) | (vals > upper)]
                    axes[i].scatter([j]*len(outliers), outliers, color='red', marker='x', s=60, label='Outlier' if i==0 and j==0 else "")
            axes[i].set_title(f'{metric} Distribution (Violin Plot)', fontsize=18)
            axes[i].set_xlabel('Strategy', fontsize=14)
            axes[i].set_ylabel(metric, fontsize=14)
            axes[i].tick_params(axis='x', rotation=20, labelsize=12)
            axes[i].grid(True, axis='y', linestyle='--', alpha=0.5)
            axes[i].set_ylim(0, 5)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'violin_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        # Figure caption: Violin plots with overlaid data points and outlier markers for each metric by strategy. Y-axis is 0-5 to show full distribution shape.
        
        # 3. Heatmap: Improvement Matrix (ordered, bold annotation, clear colorbar)
        improvement_pivot = improvements_df.pivot_table(
            values='Improvement_Percent', 
            index='Strategy', 
            columns='Metric', 
            aggfunc='mean'
        ).reindex(index=['contextual', 'instructional', 'retrieval_based'], columns=metric_order)
        plt.figure(figsize=(16, 10))
        ax = sns.heatmap(improvement_pivot, annot=True, fmt='.1f', cmap='YlGn', cbar_kws={'label': '% Improvement over Baseline'}, annot_kws={'size':16, 'weight':'bold'})
        plt.title('Improvement Matrix: Strategy vs Metric', fontsize=22)
        plt.xlabel('Metric', fontsize=16)
        plt.ylabel('Strategy', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'improvement_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        # Figure caption: Heatmap showing % improvement over baseline for each strategy and metric, ordered for clarity.
        
        # 4. Model-Strategy Heatmap (ordered, bold annotation, clear colorbar)
        model_pivot = combined_df.pivot_table(
            values='Bias_Composite', 
            index='Strategy', 
            columns='Model', 
            aggfunc='mean'
        ).reindex(index=strategy_order)
        plt.figure(figsize=(14, 10))
        ax = sns.heatmap(model_pivot, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': 'Mean Bias Composite Score'}, annot_kws={'size':16, 'weight':'bold'})
        plt.title('Bias Composite Scores: Strategy vs Model', fontsize=22)
        plt.xlabel('Model', fontsize=16)
        plt.ylabel('Strategy', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_strategy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        # Figure caption: Heatmap showing mean Bias Composite Score for each strategy and model, ordered for clarity.
        
        # 5. Radar Chart: Highlight best-performing strategy
        strategy_means = combined_df.groupby('Strategy')[metric_order[:-1]].mean().reindex(strategy_order)
        N = len(metric_order[:-1])
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        best_strategy = strategy_means.mean(axis=1).idxmax()
        for i, strategy in enumerate(strategy_order):
            values = strategy_means.loc[strategy].tolist() if strategy in strategy_means.index else [0]*N
            values += values[:1]
            lw = 4 if strategy == best_strategy else 2
            ax.plot(angles, values, 'o-', linewidth=lw, label=strategy, color=palette[i])
            ax.fill(angles, values, alpha=0.1, color=palette[i])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_order[:-1], fontsize=14)
        ax.set_ylim(0, 5)
        ax.set_title('Strategy Performance Comparison (Radar Chart)', size=18, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'strategy_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        # Figure caption: Radar chart comparing all strategies, with best-performing strategy highlighted.
        
        # 6. Add summary table with key results
        summary_data = []
        for strategy in strategy_order:
            row = {'Strategy': strategy}
            for metric in metric_order:
                row[metric] = combined_df[combined_df['Strategy'] == strategy][metric].mean()
            summary_data.append(row)
        import pandas as pd
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'summary_table.csv', index=False)
        # Figure caption: Summary table with mean scores for each strategy and metric.
        
        # 7. Box Plots: Mark outliers
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        for i, metric in enumerate(metric_order):
            sns.boxplot(data=combined_df, x='Strategy', y=metric, order=strategy_order, ax=axes[i], palette=palette, showfliers=True, flierprops={'markerfacecolor':'red', 'marker':'x', 'markersize':8})
            axes[i].set_title(f'{metric} Distribution (Box Plot)', fontsize=18)
            axes[i].set_xlabel('Strategy', fontsize=14)
            axes[i].set_ylabel(metric, fontsize=14)
            axes[i].tick_params(axis='x', rotation=20, labelsize=12)
            axes[i].grid(True, axis='y', linestyle='--', alpha=0.5)
            axes[i].set_ylim(1, 5)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'box_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        # Figure caption: Box plots with outlier markers for each metric by strategy.
    
    def create_summary_tables(self, strategy_stats, improvement_stats, model_stats):
        """Create and save summary tables."""
        
        # Save detailed statistics
        strategy_stats.to_csv(self.output_dir / 'strategy_statistics.csv')
        improvement_stats.to_csv(self.output_dir / 'improvement_statistics.csv')
        model_stats.to_csv(self.output_dir / 'model_statistics.csv')
        
        # Create simplified summary table
        summary_data = []
        for strategy in self.strategies.keys():
            strategy_data = strategy_stats.loc[strategy]
            summary_data.append({
                'Strategy': strategy,
                'Avg_Accuracy': strategy_data[('Accuracy', 'mean')],
                'Avg_Relevance': strategy_data[('Relevance', 'mean')],
                'Avg_Fairness': strategy_data[('Fairness', 'mean')],
                'Avg_Neutrality': strategy_data[('Neutrality', 'mean')],
                'Avg_Representation': strategy_data[('Representation', 'mean')],
                'Avg_Bias_Composite': strategy_data[('Bias_Composite', 'mean')]
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'summary_table.csv', index=False)
        
        return summary_df
    
    def generate_markdown_report(self, combined_df, improvements_df, summary_df):
        """Generate comprehensive markdown report with embedded plots."""
        
        report = f"""# Comprehensive Sikh Bias Mitigation Analysis Report

## Executive Summary

This report presents a comprehensive analysis of three bias mitigation strategies applied to LLM responses on Sikh-related prompts. The analysis compares baseline performance against contextual, instructional, and retrieval-based mitigation approaches.

### Key Findings

- **Total Prompts Analyzed**: {len(combined_df['Prompt ID'].unique())}
- **Models Evaluated**: {', '.join(combined_df['Model'].unique())}
- **Strategies Tested**: Baseline, Contextual, Instructional, Retrieval-Based

### Strategy Performance Overview

| Strategy | Avg Accuracy | Avg Relevance | Avg Fairness | Avg Neutrality | Avg Representation | Avg Bias Composite |
|----------|-------------|---------------|--------------|----------------|-------------------|-------------------|
"""
        
        for _, row in summary_df.iterrows():
            report += f"| {row['Strategy']} | {row['Avg_Accuracy']:.2f} | {row['Avg_Relevance']:.2f} | {row['Avg_Fairness']:.2f} | {row['Avg_Neutrality']:.2f} | {row['Avg_Representation']:.2f} | {row['Avg_Bias_Composite']:.2f} |\n"
        
        report += f"""

## Detailed Analysis

### 1. Average Scores by Strategy

![Average Scores](average_scores_by_strategy.png)

*Figure 1: Average scores across all metrics for each strategy. Higher scores indicate better performance.*

### 2. Improvement Trends

![Improvement Trends](improvement_trends.png)

*Figure 2: Percentage improvement over baseline for each strategy and metric.*

### 3. Strategy Performance Heatmap

![Improvement Heatmap](improvement_heatmap.png)

*Figure 3: Heatmap showing improvement percentages for each strategy-metric combination.*

### 4. Score Distributions

![Score Distributions](score_distributions.png)

*Figure 4: Box plots showing the distribution of scores for each strategy.*

### 5. Strategy Comparison Radar Chart

![Radar Chart](strategy_radar_chart.png)

*Figure 5: Radar chart comparing strategy performance across all metrics.*

### 6. Detailed Distributions (Violin Plots)

![Violin Plots](violin_plots.png)

*Figure 6: Violin plots showing detailed score distributions for each strategy.*

### 7. Model-Strategy Performance

![Model Strategy Heatmap](model_strategy_heatmap.png)

*Figure 7: Heatmap showing bias composite scores across different models and strategies.*

### 8. Improvement Distributions

![Improvement Distributions](improvement_distributions.png)

*Figure 8: Histograms showing the distribution of improvement percentages for each metric.*

## Statistical Analysis

### Improvement Statistics

The following table shows the average improvement percentages for each strategy:

"""
        
        # Add improvement statistics
        for metric in self.metrics + ['Bias_Composite']:
            report += f"\n#### {metric} Improvements\n"
            metric_improvements = improvements_df[improvements_df['Metric'] == metric]
            for strategy in self.strategies.keys():
                strategy_data = metric_improvements[metric_improvements['Strategy'] == strategy]
                if len(strategy_data) > 0:
                    avg_improvement = strategy_data['Improvement_Percent'].mean()
                    report += f"- **{strategy}**: {avg_improvement:.1f}% improvement\n"
        
        report += f"""

## Conclusions

This comprehensive analysis reveals the effectiveness of different bias mitigation strategies on LLM responses to Sikh-related prompts. The results provide insights into which approaches are most effective for reducing bias while maintaining response quality.

### Recommendations

Based on the analysis, the following recommendations can be made:

1. **Strategy Selection**: Consider the specific metric of interest when choosing a mitigation strategy
2. **Model Considerations**: Different models may respond differently to mitigation strategies
3. **Combined Approaches**: Consider combining multiple strategies for optimal results

## Methodology

- **Baseline Data**: Original LLM responses scored by human annotators
- **Mitigation Strategies**: Three distinct approaches applied to the same prompts
- **Metrics**: Five key dimensions plus composite bias score
- **Analysis**: Statistical comparison and visualization of improvements

---
*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(self.output_dir / 'comprehensive_analysis_report.md', 'w') as f:
            f.write(report)
        
        return report
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive mitigation analysis...")
        
        # Load data
        print("Loading baseline data...")
        baseline_df = self.load_baseline_data()
        
        print("Loading mitigated data...")
        mitigated_df = self.load_mitigated_data()
        
        # Merge data
        print("Merging baseline and mitigated data...")
        combined_df = self.merge_baseline_mitigated(baseline_df, mitigated_df)
        
        # Calculate improvements
        print("Calculating improvements...")
        improvements_df = self.calculate_improvements(combined_df)
        
        # Create summary statistics
        print("Creating summary statistics...")
        strategy_stats, improvement_stats, model_stats = self.create_summary_statistics(combined_df, improvements_df)
        
        # Generate visualizations
        print("Generating visualizations...")
        self.generate_visualizations(combined_df, improvements_df)
        
        # Create summary tables
        print("Creating summary tables...")
        summary_df = self.create_summary_tables(strategy_stats, improvement_stats, model_stats)
        
        # Generate markdown report
        print("Generating markdown report...")
        self.generate_markdown_report(combined_df, improvements_df, summary_df)
        
        # Save processed data
        print("Saving processed data...")
        combined_df.to_csv(self.output_dir / 'combined_data.csv', index=False)
        improvements_df.to_csv(self.output_dir / 'improvements_data.csv', index=False)
        
        print(f"Analysis complete! Results saved to: {self.output_dir}")
        
        return {
            'combined_df': combined_df,
            'improvements_df': improvements_df,
            'summary_df': summary_df,
            'strategy_stats': strategy_stats,
            'improvement_stats': improvement_stats,
            'model_stats': model_stats
        }

if __name__ == "__main__":
    analyzer = ComprehensiveMitigationAnalysis()
    results = analyzer.run_complete_analysis()
    
    print("\nAnalysis Summary:")
    print(f"- Total prompts analyzed: {len(results['combined_df']['Prompt ID'].unique())}")
    print(f"- Total responses: {len(results['combined_df'])}")
    print(f"- Strategies compared: {len(results['combined_df']['Strategy'].unique())}")
    print(f"- Models evaluated: {len(results['combined_df']['Model'].unique())}") 