import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from scipy import stats as scipy_stats
import warnings
from datetime import datetime
import shutil
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

MODEL_COLORS = {
    'gpt-4': '#4ECDC4',
    'claude-3-haiku-20240307': '#FF6B6B',
    'llama-3.3-70b-versatile': '#45B7D1',
}

MODEL_LABELS = {
    'gpt-4': 'GPT-4',
    'claude-3-haiku-20240307': 'Claude 3 Haiku',
    'llama-3.3-70b-versatile': 'Llama 3.3 70B',
}

# At the top, set the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPROVED_VIS_DIR = os.path.join(PROJECT_ROOT, 'visualizations', 'improved_visualizations')
ARCHIVE_OLD_VIS_DIR = os.path.join(PROJECT_ROOT, 'visualizations', 'archive', 'old_visualizations')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CSV_PATTERN = os.path.join(DATA_DIR, 'llm_sikh_bias_responses_*.csv')

# In archive_old_visualizations()
# Replace
# vis_dir = 'improved_visualizations'
# with
vis_dir = IMPROVED_VIS_DIR
# And update the archive path accordingly

def load_all_model_data(csv_pattern=CSV_PATTERN):
    """Load all model CSV files and combine them into a single dataframe"""
    # Find all CSV files matching the pattern
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        print(f"No CSV files found matching pattern: {csv_pattern}")
        return None
    all_dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            model_name = os.path.basename(csv_file).replace('llm_sikh_bias_responses_', '').replace('.csv', '')
            df['Rater_Run'] = csv_file
            all_dfs.append(df)
            print(f"Loaded {csv_file} with {len(df)} rows")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    if not all_dfs:
        return None
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined data: {len(combined_df)} total rows from {len(csv_files)} files")
    return combined_df

def archive_old_visualizations():
    """Move old PNGs from improved_visualizations/ to archive/old_visualizations/ before generating new ones."""
    archive_dir = os.path.join(ARCHIVE_OLD_VIS_DIR)
    os.makedirs(archive_dir, exist_ok=True)
    vis_dir = IMPROVED_VIS_DIR
    os.makedirs(vis_dir, exist_ok=True)  # Ensure the directory exists
    for fname in os.listdir(vis_dir):
        if fname.endswith('.png'):
            src = os.path.join(vis_dir, fname)
            dst = os.path.join(archive_dir, fname)
            shutil.move(src, dst)

# Category-based analysis with improved labeling and legend positioning
def create_category_based_analysis(df, output_dir):
    """Create category-based analysis with improved labeling"""
    # Calculate bias scores first
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    # Group by category and model
    category_model_means = df.groupby(['Category', 'Model'])['Bias_Score'].mean().reset_index()
    
    # Create subcategory analysis with category context
    subcategory_model_means = df.groupby(['Subcategory', 'Category', 'Model'])['Bias_Score'].mean().reset_index()
    
    # Create labels that include category context
    subcategory_model_means['Label'] = subcategory_model_means['Subcategory'] + ' (' + subcategory_model_means['Category'] + ')'
    
    # Sort by category and then by mean bias score
    subcategory_model_means = subcategory_model_means.sort_values(['Category', 'Bias_Score'])
    
    plt.figure(figsize=(20, 12))
    
    # Create the plot
    ax = sns.barplot(data=subcategory_model_means, x='Label', y='Bias_Score', hue='Model', 
                     palette=MODEL_COLORS, alpha=0.8)
    
    # Customize the plot
    plt.title('Bias Score by Subcategory and Model\n(Subcategory - Category)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Subcategory (Category)', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Bias Score (1-5)', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Move legend to top left
    plt.legend(title='Model', loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
    
    # Add value labels on bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.2f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontsize=8, rotation=0)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f'category_based_analysis_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create category-level analysis (simpler, less cluttered)
    plt.figure(figsize=(14, 8))
    # Prepare multi-line labels for categories
    multiline_labels = [cat.replace(' ', '\n') for cat in category_model_means['Category'].unique()]
    ax2 = sns.barplot(data=category_model_means, x='Category', y='Bias_Score', hue='Model', 
                      palette=MODEL_COLORS, alpha=0.8)
    
    plt.title('Bias Score by Category and Model', fontsize=16, fontweight='bold')
    plt.xlabel('Category', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Bias Score (1-5)', fontsize=12, fontweight='bold')
    plt.xticks(range(len(multiline_labels)), multiline_labels, rotation=0, ha='center', fontsize=11)
    plt.legend(title='Model', loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10)
    
    # Add value labels
    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
    
    plt.subplots_adjust(bottom=0.2, right=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'category_based_analysis_main_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_bias_score_ordered_analysis(df, output_dir=IMPROVED_VIS_DIR):
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

# Completely redesigned thematic progression
def create_thematic_progression(df, output_dir):
    """Create meaningful thematic progression showing bias trends across different dimensions"""
    
    # Calculate bias scores first
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    # Create multiple meaningful thematic analyses
    
    # 1. Bias Score by Prompt Complexity (based on prompt length)
    df['Prompt_Length'] = df['Prompt Text'].str.len()
    df['Complexity_Level'] = pd.cut(df['Prompt_Length'], bins=5, labels=['Very Simple', 'Simple', 'Moderate', 'Complex', 'Very Complex'])
    
    complexity_means = df.groupby(['Complexity_Level', 'Model'])['Bias_Score'].mean().reset_index()
    
    plt.figure(figsize=(14, 8))
    for model in df['Model'].unique():
        model_data = complexity_means[complexity_means['Model'] == model]
        plt.plot(range(len(model_data)), model_data['Bias_Score'], 
                marker='o', linewidth=2, markersize=8, 
                color=MODEL_COLORS.get(model, '#666666'), 
                label=MODEL_LABELS.get(model, model))
    
    plt.title('Bias Score by Prompt Complexity\n(Longer prompts = More complex)', fontsize=16, fontweight='bold')
    plt.xlabel('Prompt Complexity Level', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Bias Score (1-5)', fontsize=12, fontweight='bold')
    plt.xticks(range(5), ['Very Simple', 'Simple', 'Moderate', 'Complex', 'Very Complex'])
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f'thematic_progression_complexity_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bias Score by Category (ordered by average bias)
    category_means = df.groupby(['Category', 'Model'])['Bias_Score'].mean().reset_index()
    overall_category_means = df.groupby('Category')['Bias_Score'].mean().sort_values()
    
    plt.figure(figsize=(14, 8))
    for model in df['Model'].unique():
        model_data = category_means[category_means['Model'] == model]
        # Reorder by overall category means
        model_data = model_data.set_index('Category').reindex(overall_category_means.index).reset_index()
        plt.plot(range(len(model_data)), model_data['Bias_Score'], 
                marker='s', linewidth=2, markersize=8, 
                color=MODEL_COLORS.get(model, '#666666'), 
                label=MODEL_LABELS.get(model, model))
    
    plt.title('Bias Score by Category\n(Ordered by increasing bias across all models)', fontsize=16, fontweight='bold')
    plt.xlabel('Category (Most to Least Biased)', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Bias Score (1-5)', fontsize=12, fontweight='bold')
    plt.xticks(range(len(overall_category_means)), overall_category_means.index, rotation=45, ha='right')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'thematic_progression_category_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance consistency across categories (standard deviation)
    category_std = df.groupby(['Category', 'Model'])['Bias_Score'].std().reset_index()
    
    plt.figure(figsize=(14, 8))
    for model in df['Model'].unique():
        model_data = category_std[category_std['Model'] == model]
        plt.plot(range(len(model_data)), model_data['Bias_Score'], 
                marker='^', linewidth=2, markersize=8, 
                color=MODEL_COLORS.get(model, '#666666'), 
                label=MODEL_LABELS.get(model, model))
    
    plt.title('Bias Score Consistency by Category\n(Lower = More Consistent Performance)', fontsize=16, fontweight='bold')
    plt.xlabel('Category', fontsize=12, fontweight='bold')
    plt.ylabel('Standard Deviation of Bias Score', fontsize=12, fontweight='bold')
    plt.xticks(range(len(category_std[category_std['Model'] == df['Model'].iloc[0]])), 
               category_std[category_std['Model'] == df['Model'].iloc[0]]['Category'], 
               rotation=45, ha='right')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'thematic_progression_consistency_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_distribution_analysis(df, output_dir=IMPROVED_VIS_DIR):
    """Create distribution and box plot analysis with in-plot annotation for no-variance cases"""
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
        stds = []
        for model in models:
            model_data = df[df['Model'] == model][category].dropna()
            if len(model_data) > 0:
                data_to_plot.append(model_data)
                labels.append(f"{model}\n(μ={model_data.mean():.2f})")
                stds.append(model_data.std())
            else:
                stds.append(None)
        
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
            
            # Add in-plot annotation for no-variance cases (Claude only)
            for i, (model, std) in enumerate(zip(models, stds)):
                if 'claude' in model.lower() and std is not None and std < 0.05:
                    y = data_to_plot[i].iloc[0] if len(data_to_plot[i]) > 0 else 0
                    ax.text(i+1, y+0.1, 'No variance: all values identical',
                            ha='center', va='bottom', fontsize=10, color='gray')
            
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
    print("Note: For any Claude category with no variance, an in-plot annotation is added. Also mention this in the figure caption for publication.")

def create_heatmap_analysis(df, output_dir=IMPROVED_VIS_DIR):
    """Create both faceted and grouped heatmap analysis with improved academic formatting"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate composite scores (renamed from Bias_Score)
    df['Composite_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    # Define categories with better labeling
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Composite_Score']
    category_labels = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Composite Score']
    
    models = df['Model'].unique()
    prompt_categories = df['Category'].unique()
    
    # Create custom colormap (colorblind-friendly) -- moved to top for all heatmaps
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)

    # Create heatmap data for grouped version
    heatmap_data = []
    row_labels = []
    model_groups = []
    
    for model in models:
        for prompt_cat in prompt_categories:
            cat_model_data = df[(df['Category'] == prompt_cat) & (df['Model'] == model)]
            if len(cat_model_data) > 0:
                row = []
                for category in categories:
                    scores = cat_model_data[category].dropna()
                    row.append(scores.mean() if len(scores) > 0 else 0)
                heatmap_data.append(row)
                row_labels.append(f"{MODEL_LABELS.get(model, model)} - {prompt_cat}")
                model_groups.append(model)
    
    heatmap_df = pd.DataFrame(heatmap_data, index=row_labels, columns=category_labels)
    
    # 1. GROUPED HEATMAP (Single heatmap with grouped rows)
    # Remove floating model labels, black dividers, and yellow annotation boxes for a cleaner look
    # Optionally, add a blank row between model groups for visual separation
    cleaned_heatmap_data = []
    cleaned_row_labels = []
    last_model = None
    for i, (row, label, model) in enumerate(zip(heatmap_data, row_labels, model_groups)):
        if last_model is not None and model != last_model:
            # Insert a blank row for separation
            cleaned_heatmap_data.append([None]*len(category_labels))
            cleaned_row_labels.append('')
        cleaned_heatmap_data.append(row)
        cleaned_row_labels.append(label)
        last_model = model
    cleaned_heatmap_df = pd.DataFrame(cleaned_heatmap_data, index=cleaned_row_labels, columns=category_labels)

    plt.figure(figsize=(16, 12))
    ax = sns.heatmap(cleaned_heatmap_df, annot=True, cmap=cmap, center=3, 
                     fmt='.2f', cbar_kws={'label': 'Mean Score (1-5)'},
                     linewidths=0.5, linecolor='white',
                     mask=cleaned_heatmap_df.isnull())
    plt.title('Performance Heatmap by Model and Category (Grouped)\n(Green=Better Performance, Red=Worse Performance)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Evaluation Categories', fontsize=12, fontweight='bold')
    plt.ylabel('Model - Category Combinations', fontsize=12, fontweight='bold')
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/heatmap_analysis_grouped_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created grouped heatmap analysis: {filename}")
    
    # 2. FACETED HEATMAPS (One heatmap per model)
    fig, axes = plt.subplots(1, len(models), figsize=(20, 6))
    if len(models) == 1:
        axes = [axes]
    
    fig.suptitle('Performance Heatmaps by Model (Faceted)\n(Green=Better Performance, Red=Worse Performance)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        # Prepare data for this model
        model_data = []
        row_labels_model = []
        
        for prompt_cat in prompt_categories:
            cat_model_data = df[(df['Category'] == prompt_cat) & (df['Model'] == model)]
            if len(cat_model_data) > 0:
                row = []
                for category in categories:
                    scores = cat_model_data[category].dropna()
                    row.append(scores.mean() if len(scores) > 0 else 0)
                model_data.append(row)
                row_labels_model.append(prompt_cat)
        
        if model_data:
            model_df = pd.DataFrame(model_data, index=row_labels_model, columns=category_labels)
            
            # Create heatmap for this model
            sns.heatmap(model_df, annot=True, cmap=cmap, center=3, 
                       fmt='.2f', cbar_kws={'label': 'Mean Score (1-5)'},
                       linewidths=0.5, linecolor='white', ax=ax)
            
            ax.set_title(f'{MODEL_LABELS.get(model, model)}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Evaluation Categories', fontsize=10, fontweight='bold')
            ax.set_ylabel('Bias Categories', fontsize=10, fontweight='bold')
            
            # Rotate x-axis labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    filename = f'{output_dir}/heatmap_analysis_faceted_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created faceted heatmap analysis: {filename}")
    
    # 3. SUMMARY BAR CHART (Mean performance by model)
    plt.figure(figsize=(14, 8))
    
    # Calculate mean scores by model
    model_means = df.groupby('Model')[categories].mean()
    model_means.columns = category_labels
    
    # Create bar chart
    x = np.arange(len(category_labels))
    width = 0.25
    
    for i, model in enumerate(models):
        model_label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, '#666666')
        plt.bar(x + i*width, model_means.loc[model], width, 
               label=model_label, color=color, alpha=0.8)
    
    plt.xlabel('Evaluation Categories', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Score (1-5)', fontsize=12, fontweight='bold')
    plt.title('Mean Performance by Model and Category', fontsize=16, fontweight='bold')
    plt.xticks(x + width, category_labels, rotation=45, ha='right')
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10, title='Model')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, model in enumerate(models):
        for j, category in enumerate(category_labels):
            value = model_means.loc[model, category]
            plt.text(j + i*width, value + 0.05, f'{value:.2f}', 
                    ha='center', va='bottom', fontsize=8)
    
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    filename = f'{output_dir}/heatmap_summary_chart_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created summary chart: {filename}")

def main():
    """Main function to create improved visualizations"""
    print("CREATING IMPROVED VISUALIZATIONS")
    print("=" * 40)
    
    # Archive old visuals before generating new ones
    archive_old_visualizations()
    
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
    create_category_based_analysis(df, IMPROVED_VIS_DIR)
    
    print("Creating bias score ordered analysis...")
    create_bias_score_ordered_analysis(df)
    
    print("Creating thematic progression analysis...")
    create_thematic_progression(df, IMPROVED_VIS_DIR)
    
    print("Creating performance distribution analysis...")
    create_performance_distribution_analysis(df)
    
    print("Creating heatmap analysis...")
    create_heatmap_analysis(df)
    
    print(f"\nAll improved visualizations created in '{IMPROVED_VIS_DIR}/' directory")
    print(f"\nVisualization types created:")
    print("1. Category-based analysis - Shows performance by prompt category")
    print("2. Bias score ordered - Shows trends from low to high bias prompts")
    print("3. Thematic progression - Shows performance by topic complexity")
    print("4. Performance distribution - Box plots with statistical significance")
    print("5. Heatmap analysis - Color-coded performance matrix")

if __name__ == "__main__":
    main() 