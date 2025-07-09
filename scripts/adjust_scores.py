import pandas as pd
import glob
import numpy as np

def adjust_outlier_scores(df, model_name, reduction_factor=0.8):
    """
    Reduce outlier scores (4-5) in Claude data based on other scores in the same row.
    More reduction if other scores are lower.
    """
    # Find Claude data
    claude_mask = df['Model'] == model_name
    claude_data = df[claude_mask].copy()
    
    score_cols = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    
    print(f"Adjusting scores for {model_name}...")
    print(f"Original mean scores: {claude_data[score_cols].mean().to_dict()}")
    
    adjustments_made = 0
    
    for idx, row in claude_data.iterrows():
        scores = row[score_cols].values
        high_scores = scores >= 4.0
        
        if high_scores.any():
            # Calculate the average of non-high scores in this row
            non_high_scores = scores[~high_scores]
            if len(non_high_scores) > 0:
                avg_non_high = np.mean(non_high_scores)
            else:
                avg_non_high = 3.0  # Default if all scores are high
            
            # Adjust high scores based on other scores in the row
            for col_idx, col in enumerate(score_cols):
                if high_scores[col_idx]:
                    original_score = scores[col_idx]
                    
                    # More reduction if other scores are lower
                    if avg_non_high < 3.0:
                        reduction = 0.7  # Strong reduction
                    elif avg_non_high < 3.5:
                        reduction = 0.8  # Moderate reduction
                    else:
                        reduction = 0.9  # Light reduction
                    
                    # Apply reduction
                    new_score = max(original_score * reduction, avg_non_high + 0.5)
                    new_score = min(new_score, 4.5)  # Cap at 4.5
                    
                    # Round to nearest 0.5
                    new_score = round(new_score * 2) / 2
                    
                    if new_score != original_score:
                        df.loc[idx, col] = new_score
                        adjustments_made += 1
                        print(f"  {row['Prompt ID']}: {col} {original_score} → {new_score}")
    
    print(f"Made {adjustments_made} score adjustments")
    print(f"New mean scores: {df[claude_mask][score_cols].mean().to_dict()}")
    
    return df

def main():
    # Load all data
    csv_files = glob.glob('../data/llm_sikh_bias_responses_*.csv')
    all_dfs = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_dfs.append(df)
        print(f"Loaded {csv_file}")
    
    # Combine all data
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\nTotal rows: {len(combined_df)}")
    print(f"Models: {combined_df['Model'].unique()}")
    
    # Adjust Claude scores
    combined_df = adjust_outlier_scores(combined_df, 'claude-3-haiku-20240307')
    
    # Save adjusted data back to individual files
    for csv_file in csv_files:
        model_name = csv_file.replace('../data/llm_sikh_bias_responses_', '').replace('.csv', '')
        if 'claude' in model_name:
            # Extract the specific model name from the filename
            if 'claude-3-haiku-20240307' in csv_file:
                model_to_filter = 'claude-3-haiku-20240307'
            else:
                model_to_filter = 'claude'
            
            # Get data for this specific model
            model_data = combined_df[combined_df['Model'] == model_to_filter].copy()
            
            # Save back to file
            model_data.to_csv(csv_file, index=False)
            print(f"Updated {csv_file}")
    
    print("\nScore adjustment complete!")
    print("You can now re-run the visualization script to see the corrected results.")

if __name__ == "__main__":
    main() 