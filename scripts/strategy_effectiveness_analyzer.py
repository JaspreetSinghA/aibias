#!/usr/bin/env python3
"""
Strategy Effectiveness Analyzer

Analyzes the effectiveness of different bias mitigation strategies and provides
detailed insights for improving LLM responses to Sikh-related content.

Features:
- Comparative analysis of baseline vs. mitigated responses
- Strategy ranking and effectiveness metrics
- Statistical significance testing
- Detailed recommendations for strategy improvement
- Cost-benefit analysis of different approaches
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StrategyEffectivenessAnalyzer:
    """
    Analyzes the effectiveness of different bias mitigation strategies.
    """
    
    def __init__(self, workflow_dir: str = 'data/mitigation_workflow'):
        """Initialize the strategy effectiveness analyzer."""
        self.workflow_dir = Path(workflow_dir)
        self.strategies = ['instructional', 'contextual', 'retrieval_based']
        self.models = ['gpt_4', 'claude_3_haiku', 'llama_3.3_70b']
        
    def load_baseline_data(self) -> pd.DataFrame:
        """Load baseline response data."""
        baseline_dir = self.workflow_dir / 'semantic_similarity_strategy' / 'original_baseline_diagnostics'
        
        baseline_files = list(baseline_dir.glob('*.csv'))
        if not baseline_files:
            raise FileNotFoundError(f"No baseline files found in {baseline_dir}")
        
        baseline_data = []
        for file in baseline_files:
            df = pd.read_csv(file)
            # Extract model name from filename
            model_name = file.stem.replace('bias_analysis_baseline_', '').replace('_responses', '')
            df['model'] = model_name
            baseline_data.append(df)
        
        return pd.concat(baseline_data, ignore_index=True)
    
    def load_mitigated_data(self) -> Dict[str, pd.DataFrame]:
        """Load mitigated response data for all strategies."""
        mitigated_dir = self.workflow_dir / 'prompt_engineering_strategy' / 'bias_diagnostics'
        
        mitigated_data = {}
        for strategy in self.strategies:
            strategy_files = list(mitigated_dir.glob(f'*{strategy}*.csv'))
            if strategy_files:
                strategy_data = []
                for file in strategy_files:
                    df = pd.read_csv(file)
                    # Extract model name from filename
                    model_name = file.stem.split('_')[-1]  # Last part is model name
                    df['model'] = model_name
                    df['strategy'] = strategy
                    strategy_data.append(df)
                
                if strategy_data:
                    mitigated_data[strategy] = pd.concat(strategy_data, ignore_index=True)
        
        return mitigated_data
        

    
    def calculate_effectiveness_metrics(
        self, 
        baseline_df: pd.DataFrame, 
        mitigated_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """Calculate effectiveness metrics for each strategy."""
        
        effectiveness_metrics = {}
        
        # Baseline statistics
        baseline_avg = baseline_df['bias_similarity_score'].mean()
        baseline_std = baseline_df['bias_similarity_score'].std()
        
        for strategy, strategy_df in mitigated_data.items():
            logger.info(f"Analyzing strategy: {strategy}")
            
            # Strategy statistics
            strategy_avg = strategy_df['bias_similarity_score'].mean()
            strategy_std = strategy_df['bias_similarity_score'].std()
            
            # Improvement metrics
            improvement = baseline_avg - strategy_avg  # Lower is better
            improvement_percentage = (improvement / baseline_avg) * 100
            
            # Statistical significance
            t_stat, p_value = stats.ttest_ind(
                baseline_df['bias_similarity_score'],
                strategy_df['bias_similarity_score']
            )
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(baseline_df) - 1) * baseline_std**2 + 
                                (len(strategy_df) - 1) * strategy_std**2) / 
                               (len(baseline_df) + len(strategy_df) - 2))
            cohens_d = improvement / pooled_std
            
            # Model-specific analysis
            model_improvements = {}
            for model in self.models:
                baseline_model = baseline_df[baseline_df['model'] == model]
                strategy_model = strategy_df[strategy_df['model'] == model]
                
                if len(baseline_model) > 0 and len(strategy_model) > 0:
                    baseline_model_avg = baseline_model['bias_similarity_score'].mean()
                    strategy_model_avg = strategy_model['bias_similarity_score'].mean()
                    model_improvement = baseline_model_avg - strategy_model_avg
                    model_improvements[model] = {
                        'improvement': model_improvement,
                        'improvement_percentage': (model_improvement / baseline_model_avg) * 100,
                        'baseline_avg': baseline_model_avg,
                        'strategy_avg': strategy_model_avg
                    }
            
            effectiveness_metrics[strategy] = {
                'baseline_avg': baseline_avg,
                'strategy_avg': strategy_avg,
                'improvement': improvement,
                'improvement_percentage': improvement_percentage,
                'statistical_significance': p_value < 0.05,
                'p_value': p_value,
                't_statistic': t_stat,
                'cohens_d': cohens_d,
                'effect_size': self._interpret_effect_size(cohens_d),
                'model_improvements': model_improvements,
                'sample_size': len(strategy_df)
            }
        
        return effectiveness_metrics
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if abs(cohens_d) < 0.2:
            return "Negligible"
        elif abs(cohens_d) < 0.5:
            return "Small"
        elif abs(cohens_d) < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def rank_strategies(self, effectiveness_metrics: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        """Rank strategies by effectiveness."""
        ranked_strategies = []
        
        for strategy, metrics in effectiveness_metrics.items():
            # Calculate composite score (weighted combination of metrics)
            improvement_score = metrics['improvement_percentage']
            significance_bonus = 10 if metrics['statistical_significance'] else 0
            effect_size_score = {
                "Negligible": 0, "Small": 5, "Medium": 10, "Large": 15
            }[metrics['effect_size']]
            
            composite_score = improvement_score + significance_bonus + effect_size_score
            ranked_strategies.append((strategy, {**metrics, 'composite_score': composite_score}))
        
        # Sort by composite score (descending)
        ranked_strategies.sort(key=lambda x: x[1]['composite_score'], reverse=True)
        return ranked_strategies
    
    def generate_effectiveness_report(
        self, 
        effectiveness_metrics: Dict[str, Dict],
        ranked_strategies: List[Tuple[str, Dict]],
        output_dir: str = 'strategy_analysis'
    ) -> Dict[str, Any]:
        """Generate comprehensive effectiveness report."""
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create visualizations
        self._create_strategy_comparison_plot(effectiveness_metrics, output_dir, timestamp)
        self._create_model_effectiveness_plot(effectiveness_metrics, output_dir, timestamp)
        
        # Generate detailed report
        report_file = os.path.join(output_dir, f'strategy_effectiveness_report_{timestamp}.txt')
        self._write_effectiveness_report(effectiveness_metrics, ranked_strategies, report_file)
        
        # Save metrics to CSV
        metrics_df = self._create_metrics_dataframe(effectiveness_metrics)
        metrics_file = os.path.join(output_dir, f'strategy_metrics_{timestamp}.csv')
        metrics_df.to_csv(metrics_file, index=False)
        
        return {
            'timestamp': timestamp,
            'output_dir': output_dir,
            'report_file': report_file,
            'metrics_file': metrics_file
        }
    
    def _create_strategy_comparison_plot(
        self, 
        effectiveness_metrics: Dict[str, Dict], 
        output_dir: str, 
        timestamp: str
    ):
        """Create strategy comparison visualization."""
        
        strategies = list(effectiveness_metrics.keys())
        improvements = [metrics['improvement_percentage'] for metrics in effectiveness_metrics.values()]
        p_values = [metrics['p_value'] for metrics in effectiveness_metrics.values()]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Improvement percentage
        colors = ['#FF6B6B' if p < 0.05 else '#95A5A6' for p in p_values]
        bars = ax1.bar(strategies, improvements, color=colors, alpha=0.7)
        ax1.set_ylabel('Improvement (%)')
        ax1.set_title('Bias Reduction by Strategy')
        ax1.grid(True, alpha=0.3)
        
        # Add significance indicators
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            if p_val < 0.05:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        '*', ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        # P-values
        ax2.bar(strategies, p_values, color='#3498DB', alpha=0.7)
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
        ax2.set_ylabel('P-value')
        ax2.set_title('Statistical Significance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Effect sizes
        effect_sizes = [metrics['cohens_d'] for metrics in effectiveness_metrics.values()]
        ax3.bar(strategies, effect_sizes, color='#2ECC71', alpha=0.7)
        ax3.set_ylabel("Cohen's d")
        ax3.set_title('Effect Size')
        ax3.grid(True, alpha=0.3)
        
        # Sample sizes
        sample_sizes = [metrics['sample_size'] for metrics in effectiveness_metrics.values()]
        ax4.bar(strategies, sample_sizes, color='#F39C12', alpha=0.7)
        ax4.set_ylabel('Sample Size')
        ax4.set_title('Number of Responses')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'strategy_comparison_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_effectiveness_plot(
        self, 
        effectiveness_metrics: Dict[str, Dict], 
        output_dir: str, 
        timestamp: str
    ):
        """Create model-specific effectiveness visualization."""
        
        # Prepare data for heatmap
        strategies = list(effectiveness_metrics.keys())
        models = self.models
        
        improvement_matrix = np.zeros((len(models), len(strategies)))
        
        for i, model in enumerate(models):
            for j, strategy in enumerate(strategies):
                if model in effectiveness_metrics[strategy]['model_improvements']:
                    improvement_matrix[i, j] = effectiveness_metrics[strategy]['model_improvements'][model]['improvement_percentage']
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(improvement_matrix, 
                   xticklabels=[s.replace('_', ' ').title() for s in strategies],
                   yticklabels=[m.replace('_', ' ').title() for m in models],
                   annot=True, fmt='.1f', cmap='RdYlGn_r', center=0,
                   cbar_kws={'label': 'Improvement (%)'})
        
        plt.title('Strategy Effectiveness by Model')
        plt.xlabel('Strategy')
        plt.ylabel('Model')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'model_effectiveness_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _write_effectiveness_report(
        self, 
        effectiveness_metrics: Dict[str, Dict],
        ranked_strategies: List[Tuple[str, Dict]], 
        report_file: str
    ):
        """Write comprehensive effectiveness report."""
        
        with open(report_file, 'w') as f:
            f.write("STRATEGY EFFECTIVENESS ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total strategies analyzed: {len(effectiveness_metrics)}\n")
            f.write(f"Total models: {len(self.models)}\n\n")
            
            f.write("STRATEGY RANKING (by effectiveness)\n")
            f.write("-" * 40 + "\n")
            for i, (strategy, metrics) in enumerate(ranked_strategies, 1):
                f.write(f"\n{i}. {strategy.replace('_', ' ').title()}\n")
                f.write(f"   Improvement: {metrics['improvement_percentage']:.1f}%\n")
                f.write(f"   Statistical significance: {'Yes' if metrics['statistical_significance'] else 'No'} (p={metrics['p_value']:.4f})\n")
                f.write(f"   Effect size: {metrics['effect_size']} (Cohen's d: {metrics['cohens_d']:.3f})\n")
                f.write(f"   Sample size: {metrics['sample_size']}\n")
            
            f.write("\n\nDETAILED ANALYSIS\n")
            f.write("-" * 20 + "\n")
            for strategy, metrics in effectiveness_metrics.items():
                f.write(f"\n{strategy.replace('_', ' ').title()} Strategy:\n")
                f.write(f"  Baseline average bias score: {metrics['baseline_avg']:.3f}\n")
                f.write(f"  Strategy average bias score: {metrics['strategy_avg']:.3f}\n")
                f.write(f"  Improvement: {metrics['improvement']:.3f} ({metrics['improvement_percentage']:.1f}%)\n")
                f.write(f"  Statistical test: t={metrics['t_statistic']:.3f}, p={metrics['p_value']:.4f}\n")
                f.write(f"  Effect size: {metrics['effect_size']} (Cohen's d: {metrics['cohens_d']:.3f})\n")
                
                f.write("  Model-specific improvements:\n")
                for model, model_metrics in metrics['model_improvements'].items():
                    f.write(f"    {model}: {model_metrics['improvement_percentage']:.1f}% improvement\n")
            
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            # Generate recommendations based on results
            best_strategy = ranked_strategies[0][0] if ranked_strategies else None
            if best_strategy:
                f.write(f"1. Primary recommendation: Use {best_strategy.replace('_', ' ')} strategy\n")
            
            significant_strategies = [s for s, m in effectiveness_metrics.items() if m['statistical_significance']]
            if significant_strategies:
                f.write(f"2. Statistically significant strategies: {', '.join(significant_strategies)}\n")
            
            f.write("3. Consider model-specific optimization based on effectiveness patterns\n")
            f.write("4. Monitor long-term effectiveness and adjust strategies as needed\n")
            f.write("5. Combine multiple strategies for maximum bias reduction\n")
    
    def _create_metrics_dataframe(self, effectiveness_metrics: Dict[str, Dict]) -> pd.DataFrame:
        """Create a DataFrame with all metrics for export."""
        
        data = []
        for strategy, metrics in effectiveness_metrics.items():
            row = {
                'Strategy': strategy.replace('_', ' ').title(),
                'Baseline_Avg': metrics['baseline_avg'],
                'Strategy_Avg': metrics['strategy_avg'],
                'Improvement': metrics['improvement'],
                'Improvement_Percentage': metrics['improvement_percentage'],
                'Statistical_Significance': metrics['statistical_significance'],
                'P_Value': metrics['p_value'],
                'T_Statistic': metrics['t_statistic'],
                'Cohens_D': metrics['cohens_d'],
                'Effect_Size': metrics['effect_size'],
                'Sample_Size': metrics['sample_size']
            }
            
            # Add model-specific improvements
            for model, model_metrics in metrics['model_improvements'].items():
                row[f'{model}_Improvement_Percentage'] = model_metrics['improvement_percentage']
                row[f'{model}_Baseline_Avg'] = model_metrics['baseline_avg']
                row[f'{model}_Strategy_Avg'] = model_metrics['strategy_avg']
            
            data.append(row)
        
        return pd.DataFrame(data)


def main():
    """Main function for strategy effectiveness analysis."""
    parser = argparse.ArgumentParser(description='Strategy Effectiveness Analyzer')
    parser.add_argument('--workflow-dir', default='data/mitigation_workflow', 
                       help='Path to mitigation workflow directory')
    parser.add_argument('--output-dir', default='strategy_analysis', 
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = StrategyEffectivenessAnalyzer(args.workflow_dir)
    
    # Load data
    logger.info("Loading baseline data...")
    baseline_data = analyzer.load_baseline_data()
    
    logger.info("Loading mitigated data...")
    mitigated_data = analyzer.load_mitigated_data()
    
    # Calculate effectiveness metrics
    logger.info("Calculating effectiveness metrics...")
    effectiveness_metrics = analyzer.calculate_effectiveness_metrics(baseline_data, mitigated_data)
    
    # Rank strategies
    logger.info("Ranking strategies...")
    ranked_strategies = analyzer.rank_strategies(effectiveness_metrics)
    
    # Generate report
    logger.info("Generating effectiveness report...")
    report_data = analyzer.generate_effectiveness_report(effectiveness_metrics, ranked_strategies, args.output_dir)
    
    logger.info(f"Analysis complete! Results saved to {args.output_dir}")
    logger.info(f"Report generated: {report_data['timestamp']}")
    
    # Print top strategy
    if ranked_strategies:
        top_strategy, top_metrics = ranked_strategies[0]
        print(f"\n🏆 Top performing strategy: {top_strategy.replace('_', ' ').title()}")
        print(f"   Improvement: {top_metrics['improvement_percentage']:.1f}%")
        print(f"   Statistical significance: {'Yes' if top_metrics['statistical_significance'] else 'No'}")


if __name__ == "__main__":
    main() 