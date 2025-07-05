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

def create_category_based_analysis(df, output_dir='improved_visualizations'):
    """Create visualizations grouped by prompt categories"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Score']
    models = df['Model'].unique()
    
    # Get unique categories and subcategories
    prompt_categories = df['Category'].unique()
    prompt_subcategories = df['Subcategory'].unique()
    
    print(f"Creating category-based analysis with {len(prompt_categories)} categories")
    
    # Create category name mapping for better readability
    category_mapping = {
        'Historical Events': 'History',
        'Religious Practices': 'Religious',
        'Cultural Traditions': 'Cultural',
        'Core Beliefs': 'Beliefs',
        'Contemporary Issues': 'Contemporary',
        'Representational Bias': 'Media Bias',
        'Interfaith Relations': 'Interfaith'
    }
    
    # 1. Category-based performance
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Model Performance by Prompt Category', fontsize=18, fontweight='bold')
    
    for idx, category in enumerate(categories):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Calculate mean scores by prompt category
        category_data = []
        category_labels = []
        
        for prompt_cat in prompt_categories:
            for model in models:
                cat_model_data = df[(df['Category'] == prompt_cat) & (df['Model'] == model)][category].dropna()
                if len(cat_model_data) > 0:
                    category_data.append(cat_model_data.mean())
                    # Use shorter category names and better formatting
                    short_cat = category_mapping.get(prompt_cat, prompt_cat)
                    category_labels.append(f"{short_cat}\n{model}")
        
        # Create bar chart
        x_pos = np.arange(len(category_data))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'] * len(prompt_categories)
        bars = ax.bar(x_pos, category_data, alpha=0.7, color=colors)
        ax.set_title(f'{category} by Category', fontweight='bold', fontsize=14)
        ax.set_ylabel(f'{category} Score (1-5)', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(category_labels, rotation=0, ha='center', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 5.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, category_data)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/category_based_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created category-based analysis: {filename}")

def create_bias_score_ordered_analysis(df, output_dir='improved_visualizations'):
    """Create visualizations ordered by bias score"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Score']
    models = df['Model'].unique()
    
    # Create subplots for each category
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Model Performance Ordered by Bias Score (Low to High)', fontsize=18, fontweight='bold')
    
    for idx, category in enumerate(categories):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            
            # Sort by bias score
            sorted_data = model_data.sort_values('Bias_Score')
            scores = sorted_data[category].dropna()
            x_pos = np.arange(len(scores))
            
            # Scatter plot
            ax.scatter(x_pos, scores, 
                      label=f'{model} (mean: {scores.mean():.3f})', 
                      alpha=0.7, s=50, color=colors[i], edgecolors='black', linewidth=0.5)
            
            # Trend line
            if len(scores) > 1:
                z = np.polyfit(x_pos, scores, 1)
                p = np.poly1d(z)
                ax.plot(x_pos, p(x_pos), '-', linewidth=2, color=colors[i], alpha=0.8)
        
        ax.set_title(f'{category} (Ordered by Bias Score)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Prompt Index (Low to High Bias)', fontsize=12)
        ax.set_ylabel(f'{category} Score (1-5)', fontsize=12)
        ax.set_ylim(0.5, 5.5)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/bias_score_ordered_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created bias score ordered analysis: {filename}")

def create_thematic_progression_analysis(df, output_dir='improved_visualizations'):
    """Create visualizations with thematic progression"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    # Define thematic progression (from basic to complex topics)
    thematic_order = [
        'Historical Events',      # Basic historical knowledge
        'Religious Practices',    # Core religious concepts
        'Cultural Traditions',    # Cultural understanding
        'Core Beliefs',          # Fundamental beliefs
        'Contemporary Issues',   # Modern challenges
        'Representational Bias', # Media representation
        'Interfaith Relations'   # Complex interfaith topics
    ]
    
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Score']
    models = df['Model'].unique()
    
    # Create subplots for each category
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Model Performance by Thematic Complexity', fontsize=18, fontweight='bold')
    
    for idx, category in enumerate(categories):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            
            # Calculate mean scores for each thematic category
            thematic_scores = []
            thematic_labels = []
            
            for theme in thematic_order:
                theme_data = model_data[model_data['Category'] == theme][category].dropna()
                if len(theme_data) > 0:
                    thematic_scores.append(theme_data.mean())
                    thematic_labels.append(theme)
            
            if thematic_scores:
                x_pos = np.arange(len(thematic_scores))
                ax.plot(x_pos, thematic_scores, 'o-', linewidth=3, markersize=8,
                       label=f'{model} (mean: {np.mean(thematic_scores):.3f})', 
                       color=colors[i], alpha=0.8)
        
        ax.set_title(f'{category} by Thematic Complexity', fontweight='bold', fontsize=14)
        ax.set_xlabel('Thematic Category (Basic → Complex)', fontsize=12)
        ax.set_ylabel(f'{category} Score (1-5)', fontsize=12)
        ax.set_xticks(np.arange(len(thematic_labels)))
        ax.set_xticklabels(thematic_labels, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0.5, 5.5)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/thematic_progression_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created thematic progression analysis: {filename}")

def create_performance_distribution_analysis(df, output_dir='improved_visualizations'):
    """Create distribution and box plot analysis"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Score']
    models = df['Model'].unique()
    
    # Create comprehensive distribution analysis
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Performance Distribution Analysis by Model and Category', fontsize=18, fontweight='bold')
    
    for idx, category in enumerate(categories):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Prepare data for box plot
        data_to_plot = []
        labels = []
        
        for model in models:
            model_data = df[df['Model'] == model][category].dropna()
            if len(model_data) > 0:
                data_to_plot.append(model_data)
                labels.append(f"{model}\n(μ={model_data.mean():.2f})")
        
        if data_to_plot:
            # Create box plot
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add statistical significance
            if len(data_to_plot) > 1:
                f_stat, p_value = scipy_stats.f_oneway(*data_to_plot)
                ax.text(0.02, 0.98, f'ANOVA p={p_value:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{category} Distribution', fontweight='bold', fontsize=14)
            ax.set_ylabel('Score (1-5)', fontsize=12)
            ax.set_ylim(0.5, 5.5)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/performance_distribution_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created performance distribution analysis: {filename}")

def create_heatmap_analysis(df, output_dir='improved_visualizations'):
    """Create heatmap analysis by category and model"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias_Score']
    models = df['Model'].unique()
    prompt_categories = df['Category'].unique()
    
    # Create heatmap data
    heatmap_data = []
    row_labels = []
    
    for model in models:
        for prompt_cat in prompt_categories:
            cat_model_data = df[(df['Category'] == prompt_cat) & (df['Model'] == model)]
            if len(cat_model_data) > 0:
                row = []
                for category in categories:
                    scores = cat_model_data[category].dropna()
                    row.append(scores.mean() if len(scores) > 0 else 0)
                heatmap_data.append(row)
                row_labels.append(f"{model} - {prompt_cat}")
    
    heatmap_df = pd.DataFrame(heatmap_data, index=row_labels, columns=categories)
    
    # Create heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(heatmap_df, annot=True, cmap='RdYlGn_r', center=3, 
                fmt='.2f', cbar_kws={'label': 'Mean Score (1-5)'})
    plt.title('Performance Heatmap by Model and Category\n(Red=Higher Bias, Green=Lower Bias)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Evaluation Categories', fontsize=12)
    plt.ylabel('Model - Category Combinations', fontsize=12)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/heatmap_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created heatmap analysis: {filename}")

def main():
    """Main function to create improved visualizations"""
    print("CREATING IMPROVED VISUALIZATIONS")
    print("=" * 40)
    
    # Load data
    df = load_all_model_data()
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"\nData loaded successfully:")
    print(f"- Total rows: {len(df)}")
    print(f"- Models: {list(df['Model'].unique())}")
    print(f"- Categories: {list(df['Category'].unique())}")
    print(f"- Subcategories: {len(df['Subcategory'].unique())}")
    
    # Create improved visualizations
    print(f"\nCreating improved visualizations...")
    
    print("Creating category-based analysis...")
    create_category_based_analysis(df)
    
    print("Creating bias score ordered analysis...")
    create_bias_score_ordered_analysis(df)
    
    print("Creating thematic progression analysis...")
    create_thematic_progression_analysis(df)
    
    print("Creating performance distribution analysis...")
    create_performance_distribution_analysis(df)
    
    print("Creating heatmap analysis...")
    create_heatmap_analysis(df)
    
    print(f"\nAll improved visualizations created in 'improved_visualizations/' directory")
    print(f"\nVisualization types created:")
    print("1. Category-based analysis - Shows performance by prompt category")
    print("2. Bias score ordered - Shows trends from low to high bias prompts")
    print("3. Thematic progression - Shows performance by topic complexity")
    print("4. Performance distribution - Box plots with statistical significance")
    print("5. Heatmap analysis - Color-coded performance matrix")

if __name__ == "__main__":
    main() 