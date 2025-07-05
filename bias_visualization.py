import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_all_model_data(csv_pattern='llm_sikh_bias_responses_*.csv'):
    """
    Load all model CSV files and combine them into a single dataframe
    Supports multiple CSVs per model (e.g., different raters or runs)
    """
    all_data = []
    
    # Find all CSV files matching the pattern
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {csv_pattern}")
        return None
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Extract model name from filename
            model_name = csv_file.replace('llm_sikh_bias_responses_', '').replace('.csv', '')
            
            # Add model column if it doesn't exist
            if 'Model' not in df.columns:
                df['Model'] = model_name
            
            # Add rater/run identifier if multiple files per model
            df['Rater_Run'] = csv_file
            
            all_data.append(df)
            print(f"Loaded {csv_file} with {len(df)} rows")
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_data:
        print("No data loaded successfully")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined data: {len(combined_df)} total rows from {len(csv_files)} files")
    
    return combined_df

def calculate_bias_scores(df):
    """
    Calculate bias scores and comprehensive statistics
    """
    # Calculate individual bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    # Calculate model-level aggregations with confidence intervals
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Score']
    
    model_stats = {}
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        stats_dict = {}
        
        for category in categories:
            scores = model_data[category].dropna()
            if len(scores) > 0:
                stats_dict[category] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'sem': scores.sem(),  # Standard error of mean
                    'count': len(scores),
                    'min': scores.min(),
                    'max': scores.max(),
                    'median': scores.median()
                }
        
        model_stats[model] = stats_dict
    
    return model_stats

def create_radar_chart(df, output_dir='bias_analysis_plots'):
    """
    Create radar chart showing model performance across all dimensions
    """
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    # Get mean scores for each model and category
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    models = df['Model'].unique()
    
    # Number of variables
    N = len(categories)
    
    # Create angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each model
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        values = []
        
        for category in categories:
            scores = model_data[category].dropna()
            values.append(scores.mean() if len(scores) > 0 else 0)
        
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Model Performance Radar Chart\n(1=Low, 5=High)', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created radar_chart.png")

def create_heatmap(df, output_dir='bias_analysis_plots'):
    """
    Create heatmap showing bias concentration by category and model
    """
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    # Create pivot table for heatmap
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Score']
    
    heatmap_data = []
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        row = []
        for category in categories:
            scores = model_data[category].dropna()
            row.append(scores.mean() if len(scores) > 0 else 0)
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                             index=df['Model'].unique(), 
                             columns=categories)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_df, annot=True, cmap='RdYlGn_r', center=3, 
                fmt='.2f', cbar_kws={'label': 'Mean Score (1-5)'})
    plt.title('Bias Analysis Heatmap\n(Red=Higher Bias, Green=Lower Bias)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Evaluation Categories', fontsize=12)
    plt.ylabel('AI Models', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bias_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created bias_heatmap.png")

def create_category_comparison_enhanced(df, output_dir='bias_analysis_plots'):
    """
    Enhanced category comparison with statistical significance testing
    """
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Score']
    models = df['Model'].unique()
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Detailed Category Analysis with Statistical Significance', 
                 fontsize=16, fontweight='bold')
    
    for idx, category in enumerate(categories):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Prepare data for boxplot
        data_to_plot = []
        labels = []
        
        for model in models:
            model_data = df[df['Model'] == model][category].dropna()
            if len(model_data) > 0:
                data_to_plot.append(model_data)
                labels.append(model)
        
        if data_to_plot:
            # Create boxplot
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            # Add statistical significance indicators
            if len(data_to_plot) > 1:
                # Perform ANOVA
                f_stat, p_value = scipy_stats.f_oneway(*data_to_plot)
                ax.text(0.02, 0.98, f'ANOVA p={p_value:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{category} Distribution', fontweight='bold')
            ax.set_ylabel('Score (1-5)')
            ax.set_ylim(0.5, 5.5)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/enhanced_category_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created enhanced_category_comparison.png")

def create_trend_analysis(df, output_dir='bias_analysis_plots'):
    """
    Create trend analysis plots with prompt IDs and category grouping
    """
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Score']
    models = df['Model'].unique()
    
    # Create subplots for each category
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Trend Analysis by Prompt ID and Category', fontsize=16, fontweight='bold')
    
    for idx, category in enumerate(categories):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Get unique prompts and sort them
        all_prompts = df['Prompt ID'].unique()
        all_prompts = sorted(all_prompts)
        
        # Create x-axis positions
        x_positions = np.arange(len(all_prompts))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
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
                
                # Scatter plot
                ax.scatter(valid_x, valid_scores, 
                          label=f'{model}', alpha=0.7, s=50, color=colors[i])
                
                # Trend line
                if len(valid_scores) > 1:
                    z = np.polyfit(valid_x, valid_scores, 1)
                    p = np.poly1d(z)
                    ax.plot(valid_x, p(valid_x), '-', linewidth=2, 
                           color=colors[i], alpha=0.8)
        
        ax.set_title(f'{category} Trends', fontweight='bold')
        ax.set_xlabel('Prompt Index')
        ax.set_ylabel('Score (1-5)')
        ax.set_ylim(0.5, 5.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels (show every nth label to avoid crowding)
        n_labels = min(10, len(all_prompts))
        step = max(1, len(all_prompts) // n_labels)
        ax.set_xticks(x_positions[::step])
        ax.set_xticklabels([all_prompts[i] for i in range(0, len(all_prompts), step)], 
                          rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/trend_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created trend_analysis.png")

def create_statistical_summary(df, output_dir='bias_analysis_plots'):
    """
    Create comprehensive statistical summary table and visualization
    """
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    # Calculate comprehensive statistics
    model_stats = calculate_bias_scores(df)
    
    # Create summary table
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Score']
    
    summary_data = []
    for model in model_stats:
        row = [model]
        for category in categories:
            if category in model_stats[model]:
                stats = model_stats[model][category]
                row.extend([stats['mean'], stats['std'], stats['count']])
            else:
                row.extend([0, 0, 0])
        summary_data.append(row)
    
    # Create column names
    columns = ['Model']
    for category in categories:
        columns.extend([f'{category}_Mean', f'{category}_Std', f'{category}_Count'])
    
    summary_df = pd.DataFrame(summary_data, columns=columns)
    
    # Save summary table
    summary_df.to_csv(f'{output_dir}/statistical_summary.csv', index=False)
    
    # Create visualization of summary statistics
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Statistical Summary by Model and Category', fontsize=16, fontweight='bold')
    
    for idx, category in enumerate(categories):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        models = []
        means = []
        stds = []
        
        for model in model_stats:
            if category in model_stats[model]:
                models.append(model)
                means.append(model_stats[model][category]['mean'])
                stds.append(model_stats[model][category]['std'])
        
        if models:
            x_pos = np.arange(len(models))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_title(f'{category} - Mean ± Std', fontweight='bold')
            ax.set_ylabel('Score (1-5)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylim(0, 5.5)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/statistical_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created statistical_summary.csv and statistical_summary.png")
    return summary_df

def generate_research_report(df, output_dir='bias_analysis_plots'):
    """
    Generate comprehensive research report suitable for academic publication
    """
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    # Calculate statistics
    model_stats = calculate_bias_scores(df)
    
    # Create report
    report_file = f'{output_dir}/research_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("AI BIAS IN SIKH REPRESENTATION - RESEARCH REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total responses analyzed: {len(df)}\n")
        f.write(f"Models evaluated: {len(df['Model'].unique())}\n")
        f.write(f"Prompts per model: {len(df['Prompt ID'].unique())}\n")
        f.write(f"Evaluation categories: 5 (Accuracy, Relevance, Fairness, Neutrality, Representation)\n\n")
        
        f.write("KEY FINDINGS\n")
        f.write("-" * 15 + "\n")
        
        # Overall performance
        overall_bias = df['Bias_Score'].mean()
        f.write(f"Overall bias score across all models: {overall_bias:.3f}\n")
        
        # Model comparison
        for model in model_stats:
            if 'Bias_Score' in model_stats[model]:
                bias_score = model_stats[model]['Bias_Score']['mean']
                f.write(f"{model}: Bias Score = {bias_score:.3f}\n")
        
        f.write("\n\nDETAILED ANALYSIS\n")
        f.write("-" * 20 + "\n")
        
        for model in model_stats:
            f.write(f"\n{model.upper()}:\n")
            f.write(f"  Total responses: {len(df[df['Model'] == model])}\n")
            
            for category in ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Score']:
                if category in model_stats[model]:
                    stats = model_stats[model][category]
                    f.write(f"  {category}: Mean={stats['mean']:.3f}, Std={stats['std']:.3f}, ")
                    f.write(f"N={stats['count']}, Range=[{stats['min']:.3f}, {stats['max']:.3f}]\n")
        
        f.write("\n\nSTATISTICAL SIGNIFICANCE\n")
        f.write("-" * 25 + "\n")
        
        # Perform statistical tests
        categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Score']
        models = list(df['Model'].unique())
        
        if len(models) > 1:
            for category in categories:
                groups = [df[df['Model'] == model][category].dropna() for model in models]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) > 1:
                    f_stat, p_value = scipy_stats.f_oneway(*groups)
                    f.write(f"{category}: F={f_stat:.3f}, p={p_value:.3f}")
                    if p_value < 0.05:
                        f.write(" (Significant difference between models)\n")
                    else:
                        f.write(" (No significant difference)\n")
        
        f.write("\n\nRECOMMENDATIONS\n")
        f.write("-" * 15 + "\n")
        f.write("1. Models with lower bias scores require targeted improvements\n")
        f.write("2. Focus on category-specific weaknesses identified in analysis\n")
        f.write("3. Implement regular bias testing protocols\n")
        f.write("4. Engage with Sikh community for feedback and validation\n")
        f.write("5. Consider fine-tuning on diverse religious content\n")
    
    print(f"Generated comprehensive research report: {report_file}")

def main():
    """
    Main function to run the complete research-optimized bias analysis
    """
    print("RESEARCH-OPTIMIZED BIAS ANALYSIS TOOL")
    print("=" * 50)
    
    # Load all model data
    df = load_all_model_data()
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create output directory
    output_dir = 'bias_analysis_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate all visualizations
    print("\nGenerating research-optimized visualizations...")
    
    print("Creating radar chart...")
    create_radar_chart(df, output_dir)
    
    print("Creating bias heatmap...")
    create_heatmap(df, output_dir)
    
    print("Creating enhanced category comparison...")
    create_category_comparison_enhanced(df, output_dir)
    
    print("Creating trend analysis...")
    create_trend_analysis(df, output_dir)
    
    print("Creating statistical summary...")
    summary_df = create_statistical_summary(df, output_dir)
    
    print("Generating research report...")
    generate_research_report(df, output_dir)
    
    print(f"\nAnalysis complete! All files saved to '{output_dir}/' directory")
    print("\nGenerated files:")
    print("- radar_chart.png: Model performance comparison")
    print("- bias_heatmap.png: Bias concentration visualization")
    print("- enhanced_category_comparison.png: Statistical significance testing")
    print("- trend_analysis.png: Performance trends by prompt")
    print("- statistical_summary.png: Comprehensive statistics")
    print("- statistical_summary.csv: Raw statistical data")
    print("- research_report.txt: Academic publication report")
    
    print("\nSummary of average scores:")
    print(summary_df[['Model'] + [col for col in summary_df.columns if col.endswith('_Mean')]])

if __name__ == "__main__":
    main() 