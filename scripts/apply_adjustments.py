#!/usr/bin/env python3
"""
Apply Score Adjustments for Sikh Bias Analysis
Applies adjustments identified in the adjustment report:
- Sets model scores equal to rater scores when model score > rater score
- Does NOT prune outliers (preserves potentially legitimate high scores)
"""

import pandas as pd
import os
from datetime import datetime

def load_and_clean_data(file_path):
    """Load CSV and handle missing values"""
    try:
        df = pd.read_csv(file_path)
        # Convert score columns to numeric, handling missing values
        score_columns = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
        for col in score_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def apply_adjustments(rater_df, model_df, rater_name, model_name):
    """Apply score adjustments based on rater scores"""
    adjustments_made = []
    
    # Score columns to compare
    score_columns = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    
    # Create a copy of the model dataframe to modify
    adjusted_df = model_df.copy()
    
    # Merge on Prompt ID to compare scores
    merged = pd.merge(rater_df, adjusted_df, on='Prompt ID', suffixes=('_rater', '_model'))
    
    for idx, row in merged.iterrows():
        prompt_id = row['Prompt ID']
        prompt_adjustments = []
        
        # Check each score column
        for col in score_columns:
            rater_col = f"{col}_rater"
            model_col = f"{col}_model"
            
            if rater_col in row and model_col in row:
                rater_score = row[rater_col]
                model_score = row[model_col]
                
                # Skip if either score is missing
                if pd.isna(rater_score) or pd.isna(model_score):
                    continue
                
                # Apply adjustment if model score > rater score
                if model_score > rater_score:
                    # Find the corresponding row in the adjusted dataframe
                    model_idx = adjusted_df[adjusted_df['Prompt ID'] == prompt_id].index
                    if len(model_idx) > 0:
                        adjusted_df.loc[model_idx[0], col] = rater_score
                        prompt_adjustments.append({
                            'category': col,
                            'old_score': model_score,
                            'new_score': rater_score
                        })
        
        if prompt_adjustments:
            adjustments_made.append({
                'prompt_id': prompt_id,
                'adjustments': prompt_adjustments
            })
    
    return adjusted_df, adjustments_made

def apply_all_adjustments():
    """Apply adjustments to all model files"""
    
    # File mappings
    file_mappings = [
        {
            'rater_file': 'archive/Sikh Biases LLM - Narveer - LLM#2.csv',
            'model_file': 'llm_sikh_bias_responses_Narveer_claude-3-haiku-20240307.csv',
            'rater_name': 'Narveer',
            'model_name': 'claude-3-haiku-20240307'
        },
        {
            'rater_file': 'archive/Sikh Biases LLM - Jaspreet - LLM#2.csv',
            'model_file': 'llm_sikh_bias_responses_Jaspreet_claude-3-haiku-20240307.csv',
            'rater_name': 'Jaspreet',
            'model_name': 'claude-3-haiku-20240307'
        }
    ]
    
    print("=" * 80)
    print("APPLYING SCORE ADJUSTMENTS")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    total_adjustments = 0
    
    for mapping in file_mappings:
        print(f"Processing: {mapping['rater_name']} → {mapping['model_name']}")
        print("-" * 60)
        
        # Load data
        rater_df = load_and_clean_data(mapping['rater_file'])
        model_df = load_and_clean_data(mapping['model_file'])
        
        if rater_df is None or model_df is None:
            print(f"ERROR: Could not load files for {mapping['rater_name']}")
            continue
        
        # Apply adjustments
        adjusted_df, adjustments = apply_adjustments(
            rater_df, model_df, 
            mapping['rater_name'], mapping['model_name']
        )
        
        # Save adjusted file
        output_file = mapping['model_file']
        adjusted_df.to_csv(output_file, index=False)
        
        print(f"Applied {len(adjustments)} prompt adjustments")
        total_adjustments += len(adjustments)
        
        # Show some examples of adjustments
        if adjustments:
            print("Sample adjustments:")
            for i, adj in enumerate(adjustments[:3]):  # Show first 3
                print(f"  {adj['prompt_id']}:")
                for change in adj['adjustments']:
                    print(f"    {change['category']}: {change['old_score']} → {change['new_score']}")
            if len(adjustments) > 3:
                print(f"  ... and {len(adjustments) - 3} more")
        print(f"Saved adjusted file: {output_file}")
        print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total adjustments applied: {total_adjustments}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Note: Outliers were NOT pruned to preserve potentially legitimate high scores.")
    print("=" * 80)

if __name__ == "__main__":
    apply_all_adjustments() 