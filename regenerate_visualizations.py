import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from scipy import stats as scipy_stats
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_all_model_data(csv_pattern='llm_sikh_bias_responses_*.csv'):
    """Load all model CSV files and combine them"""
    all_data = []
    csv_files = glob.glob(csv_pattern)
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            model_name = csv_file.replace('llm_sikh_bias_responses_', '').replace('.csv', '')
            if 'Model' not in df.columns:
                df['Model'] = model_name
            all_data.append(df)
            print(f"Loaded {csv_file} with {len(df)} rows")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_data:
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined data: {len(combined_df)} total rows from {len(csv_files)} files")
    return combined_df

def create_enhanced_trend_analysis(df, output_dir='fresh_visualizations'):
    """Create enhanced trend analysis with all models and all datapoints"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Score']
    models = df['Model'].unique()
    
    print(f"Creating trend analysis with {len(models)} models and {len(df['Prompt ID'].unique())} prompts")
    
    # Create subplots for each category
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Enhanced Trend Analysis: All Models and All 55 Prompts', fontsize=18, fontweight='bold')
    
    for idx, category in enumerate(categories):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Get unique prompts and sort them
        all_prompts = sorted(df['Prompt ID'].unique())
        x_positions = np.arange(len(all_prompts))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Distinct colors for each model
        
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            
            # Get scores for each prompt
            scores = []
            for prompt in all_prompts:
                prompt_data = model_data[model_data['Prompt ID'] == prompt][category].dropna()
                if len(prompt_data) > 0:
                    scores.append(prompt_data.mean())
                else:
                    scores.append(np.nan)
            
            # Plot with trend line
            valid_mask = ~np.isnan(scores)
            if np.sum(valid_mask) > 1:
                valid_x = x_positions[valid_mask]
                valid_scores = np.array(scores)[valid_mask]
                
                # Scatter plot with larger points
                ax.scatter(valid_x, valid_scores, 
                          label=f'{model} ({len(valid_scores)} points)', 
                          alpha=0.8, s=80, color=colors[i], edgecolors='black', linewidth=0.5)
                
                # Trend line
                if len(valid_scores) > 1:
                    z = np.polyfit(valid_x, valid_scores, 1)
                    p = np.poly1d(z)
                    ax.plot(valid_x, p(valid_x), '-', linewidth=3, 
                           color=colors[i], alpha=0.9)
        
        ax.set_title(f'{category} Trends (All 55 Prompts)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Prompt Index (H1-H55)', fontsize=12)
        ax.set_ylabel(f'{category} Score (1-5)', fontsize=12)
        ax.set_ylim(0.5, 5.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels (show every 5th label to avoid crowding)
        step = max(1, len(all_prompts) // 10)
        ax.set_xticks(x_positions[::step])
        ax.set_xticklabels([all_prompts[i] for i in range(0, len(all_prompts), step)], 
                          rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/enhanced_trend_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created enhanced trend analysis: {filename}")

def create_model_comparison_plots(df, output_dir='fresh_visualizations'):
    """Create individual model comparison plots"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Score']
    models = df['Model'].unique()
    
    # Create individual plots for each category
    for category in categories:
        plt.figure(figsize=(16, 10))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            scores = model_data[category].dropna()
            x_pos = np.arange(len(scores))
            
            plt.scatter(x_pos, scores, 
                       label=f'{model} (mean: {scores.mean():.3f})', 
                       alpha=0.7, s=60, color=colors[i], edgecolors='black', linewidth=0.5)
            
            # Add trend line
            if len(scores) > 1:
                z = np.polyfit(x_pos, scores, 1)
                p = np.poly1d(z)
                plt.plot(x_pos, p(x_pos), '-', linewidth=2, color=colors[i], alpha=0.8)
        
        plt.title(f'{category} Scores by Model (All 55 Prompts)', fontsize=16, fontweight='bold')
        plt.xlabel('Prompt Index (1-55)', fontsize=12)
        plt.ylabel(f'{category} Score (1-5)', fontsize=12)
        plt.ylim(0.5, 5.5)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{output_dir}/{category}_model_comparison_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created {category} comparison: {filename}")

def create_summary_statistics(df, output_dir='fresh_visualizations'):
    """Create summary statistics with all models"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Score']
    models = df['Model'].unique()
    
    # Create summary table
    summary_data = []
    for model in models:
        model_data = df[df['Model'] == model]
        row = [model]
        for category in categories:
            scores = model_data[category].dropna()
            row.extend([scores.mean(), scores.std(), len(scores)])
        summary_data.append(row)
    
    columns = ['Model']
    for category in categories:
        columns.extend([f'{category}_Mean', f'{category}_Std', f'{category}_Count'])
    
    summary_df = pd.DataFrame(summary_data, columns=columns)
    
    # Save summary table
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f'{output_dir}/summary_statistics_{timestamp}.csv'
    summary_df.to_csv(summary_file, index=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Summary Statistics: All Models and Categories', fontsize=16, fontweight='bold')
    
    for idx, category in enumerate(categories):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        means = []
        stds = []
        model_names = []
        
        for model in models:
            model_data = df[df['Model'] == model]
            scores = model_data[category].dropna()
            means.append(scores.mean())
            stds.append(scores.std())
            model_names.append(model)
        
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_title(f'{category} - Mean ± Std', fontweight='bold')
        ax.set_ylabel('Score (1-5)', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(0, 5.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/summary_statistics_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created summary statistics: {summary_file} and {filename}")
    return summary_df

def main():
    """Main function to regenerate all visualizations"""
    print("REGENERATING FRESH VISUALIZATIONS")
    print("=" * 40)
    
    # Load data
    df = load_all_model_data()
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"\nData loaded successfully:")
    print(f"- Total rows: {len(df)}")
    print(f"- Models: {list(df['Model'].unique())}")
    print(f"- Prompts per model: {len(df['Prompt ID'].unique())}")
    print(f"- Categories: {len(['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation'])}")
    
    # Create fresh visualizations
    print(f"\nCreating fresh visualizations...")
    
    print("Creating enhanced trend analysis...")
    create_enhanced_trend_analysis(df)
    
    print("Creating model comparison plots...")
    create_model_comparison_plots(df)
    
    print("Creating summary statistics...")
    summary_df = create_summary_statistics(df)
    
    print(f"\nAll visualizations created in 'fresh_visualizations/' directory")
    print(f"Summary of results:")
    print(summary_df[['Model'] + [col for col in summary_df.columns if col.endswith('_Mean')]])

if __name__ == "__main__":
    main() 