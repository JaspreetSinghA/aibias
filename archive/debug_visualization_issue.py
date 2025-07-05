import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

def debug_visualization_data():
    """Debug the visualization data issue"""
    print("DEBUGGING VISUALIZATION DATA")
    print("=" * 40)
    
    # Load all data
    csv_files = glob.glob('llm_sikh_bias_responses_*.csv')
    all_data = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        model_name = csv_file.replace('llm_sikh_bias_responses_', '').replace('.csv', '')
        if 'Model' not in df.columns:
            df['Model'] = model_name
        all_data.append(df)
        print(f"Loaded {csv_file}: {len(df)} rows, model: {model_name}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\nCombined data: {len(combined_df)} total rows")
    print(f"Models: {combined_df['Model'].unique()}")
    print(f"Unique prompts: {len(combined_df['Prompt ID'].unique())}")
    
    # Check data by model
    for model in combined_df['Model'].unique():
        model_data = combined_df[combined_df['Model'] == model]
        print(f"\n{model}:")
        print(f"  Rows: {len(model_data)}")
        print(f"  Unique prompts: {len(model_data['Prompt ID'].unique())}")
        print(f"  Prompt IDs: {sorted(model_data['Prompt ID'].unique())[:10]}...")  # Show first 10
    
    # Test trend analysis manually
    print(f"\nTESTING TREND ANALYSIS")
    print("-" * 30)
    
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    models = combined_df['Model'].unique()
    
    for category in categories:
        print(f"\n{category}:")
        for model in models:
            model_data = combined_df[combined_df['Model'] == model]
            scores = model_data[category].dropna()
            print(f"  {model}: {len(scores)} scores, mean: {scores.mean():.3f}")
    
    # Create a simple test plot
    print(f"\nCREATING TEST PLOT")
    print("-" * 20)
    
    plt.figure(figsize=(12, 8))
    
    for i, model in enumerate(models):
        model_data = combined_df[combined_df['Model'] == model]
        scores = model_data['Accuracy'].dropna()
        x_pos = np.arange(len(scores))
        plt.scatter(x_pos, scores, label=model, alpha=0.7, s=30)
    
    plt.title('Test Plot: Accuracy Scores by Model')
    plt.xlabel('Response Index')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('debug_test_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created debug_test_plot.png")
    
    # Check if there are any data issues
    print(f"\nDATA QUALITY CHECK")
    print("-" * 20)
    
    for model in models:
        model_data = combined_df[combined_df['Model'] == model]
        print(f"\n{model}:")
        for category in categories:
            scores = model_data[category].dropna()
            print(f"  {category}: {len(scores)} valid scores, range: [{scores.min():.1f}, {scores.max():.1f}]")
    
    return combined_df

if __name__ == "__main__":
    df = debug_visualization_data() 