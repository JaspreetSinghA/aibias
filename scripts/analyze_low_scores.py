import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def analyze_low_scores():
    """Analyze the distribution of low scores (1s and 2s) across all models and prompts."""
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and 'llm_sikh_bias_responses' in f and f != 'llm_sikh_bias_responses.csv']
    
    print('Comprehensive Analysis of Low Scores (1s and 2s)')
    print('=' * 60)
    
    # Load all data
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
        df['source_file'] = csv_file
        # Extract model name from filename
        model_name = csv_file.replace('llm_sikh_bias_responses_', '').replace('.csv', '')
        df['model'] = model_name
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Scoring categories
    scoring_cols = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    
    print('\n1. OVERALL LOW SCORE DISTRIBUTION BY MODEL')
    print('-' * 50)
    
    model_stats = {}
    for model in combined_df['model'].unique():
        model_data = combined_df[combined_df['model'] == model]
        model_stats[model] = {}
        
        print(f'\nModel: {model}')
        print(f'Total prompts: {len(model_data)}')
        
        for col in scoring_cols:
            if col in model_data.columns:
                ones = (model_data[col] == 1).sum()
                twos = (model_data[col] == 2).sum()
                total = len(model_data)
                
                model_stats[model][col] = {
                    'ones': ones,
                    'twos': twos,
                    'total_low': ones + twos,
                    'percentage_low': (ones + twos) / total * 100
                }
                
                print(f'  {col}:')
                print(f'    1s: {ones} ({ones/total*100:.1f}%)')
                print(f'    2s: {twos} ({twos/total*100:.1f}%)')
                print(f'    Total low: {ones + twos} ({(ones + twos)/total*100:.1f}%)')
    
    print('\n2. PROMPTS WITH HIGHEST LOW SCORE PERCENTAGES')
    print('-' * 50)
    
    prompt_low_scores = {}
    for prompt_id in combined_df['Prompt ID'].unique():
        prompt_data = combined_df[combined_df['Prompt ID'] == prompt_id]
        
        total_low_scores = 0
        total_ratings = 0
        
        for col in scoring_cols:
            if col in prompt_data.columns:
                low_scores = (prompt_data[col].isin([1, 2])).sum()
                total_low_scores += low_scores
                total_ratings += len(prompt_data)
        
        if total_ratings > 0:
            percentage = (total_low_scores / total_ratings) * 100
            prompt_low_scores[prompt_id] = {
                'total_low': total_low_scores,
                'total_ratings': total_ratings,
                'percentage': percentage,
                'text': prompt_data['Prompt Text'].iloc[0]
            }
    
    # Sort by percentage of low scores
    sorted_prompts = sorted(prompt_low_scores.items(), key=lambda x: x[1]['percentage'], reverse=True)
    
    print(f'Top 15 prompts with highest percentage of low scores:')
    for i, (prompt_id, stats) in enumerate(sorted_prompts[:15]):
        print(f'\n{i+1}. Prompt ID: {prompt_id}')
        print(f'   Low scores: {stats["total_low"]}/{stats["total_ratings"]} ({stats["percentage"]:.1f}%)')
        print(f'   Text: {stats["text"][:100]}...' if len(stats['text']) > 100 else f'   Text: {stats["text"]}')
    
    print('\n3. CATEGORY-SPECIFIC LOW SCORE ANALYSIS')
    print('-' * 50)
    
    category_stats = {}
    for col in scoring_cols:
        if col in combined_df.columns:
            ones = (combined_df[col] == 1).sum()
            twos = (combined_df[col] == 2).sum()
            total = len(combined_df)
            
            category_stats[col] = {
                'ones': ones,
                'twos': twos,
                'total_low': ones + twos,
                'percentage_low': (ones + twos) / total * 100
            }
            
            print(f'\n{col}:')
            print(f'  1s: {ones} ({ones/total*100:.1f}%)')
            print(f'  2s: {twos} ({twos/total*100:.1f}%)')
            print(f'  Total low: {ones + twos} ({(ones + twos)/total*100:.1f}%)')
    
    print('\n4. PROMPTS WITH CONSISTENT LOW SCORES ACROSS MULTIPLE RATERS')
    print('-' * 50)
    
    consistent_low_prompts = []
    for prompt_id in combined_df['Prompt ID'].unique():
        prompt_data = combined_df[combined_df['Prompt ID'] == prompt_id]
        
        # Count raters who gave multiple low scores
        low_score_raters = 0
        for _, row in prompt_data.iterrows():
            low_scores = sum(1 for col in scoring_cols if col in row and row[col] in [1, 2])
            if low_scores >= 3:  # At least 3 low scores
                low_score_raters += 1
        
        if low_score_raters >= 2:  # At least 2 raters gave multiple low scores
            consistent_low_prompts.append({
                'prompt_id': prompt_id,
                'low_score_raters': low_score_raters,
                'total_raters': len(prompt_data),
                'text': prompt_data['Prompt Text'].iloc[0]
            })
    
    # Sort by number of raters with low scores
    consistent_low_prompts.sort(key=lambda x: x['low_score_raters'], reverse=True)
    
    print(f'Found {len(consistent_low_prompts)} prompts with consistent low scores:')
    for prompt in consistent_low_prompts[:10]:
        print(f'\nPrompt ID: {prompt["prompt_id"]}')
        print(f'Raters with multiple low scores: {prompt["low_score_raters"]}/{prompt["total_raters"]}')
        print(f'Text: {prompt["text"][:100]}...' if len(prompt['text']) > 100 else f'Text: {prompt["text"]}')
    
    print('\n5. SUMMARY STATISTICS')
    print('-' * 50)
    
    total_prompts = len(combined_df['Prompt ID'].unique())
    total_ratings = len(combined_df)
    
    print(f'Total unique prompts: {total_prompts}')
    print(f'Total ratings across all models: {total_ratings}')
    
    # Overall low score statistics
    total_low_scores = sum(stats['total_low'] for stats in category_stats.values())
    overall_low_percentage = (total_low_scores / (total_ratings * len(scoring_cols))) * 100
    
    print(f'Total low scores (1s + 2s): {total_low_scores}')
    print(f'Overall low score percentage: {overall_low_percentage:.1f}%')
    
    # Model comparison
    print(f'\nModel comparison (low score percentages):')
    for model, stats in model_stats.items():
        avg_low_percentage = sum(cat_stats['percentage_low'] for cat_stats in stats.values()) / len(stats)
        print(f'  {model}: {avg_low_percentage:.1f}%')
    
    return {
        'model_stats': model_stats,
        'category_stats': category_stats,
        'prompt_low_scores': prompt_low_scores,
        'consistent_low_prompts': consistent_low_prompts
    }

if __name__ == "__main__":
    results = analyze_low_scores() 