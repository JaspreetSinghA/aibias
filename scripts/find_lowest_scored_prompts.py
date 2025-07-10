#!/usr/bin/env python3
"""
Script to find the 5 lowest scored prompts based on average bias score across models.
Bias score = (Neutrality + Representation + Fairness) / 3
"""

import pandas as pd
import os
import glob
from pathlib import Path

def calculate_bias_score(row):
    """Calculate bias score from Neutrality, Representation, and Fairness scores."""
    try:
        neutrality = float(row['Neutrality']) if pd.notna(row['Neutrality']) else 0
        representation = float(row['Representation']) if pd.notna(row['Representation']) else 0
        fairness = float(row['Fairness']) if pd.notna(row['Fairness']) else 0
        
        # Calculate bias score
        bias_score = (neutrality + representation + fairness) / 3
        return bias_score
    except (ValueError, TypeError):
        return None

def find_lowest_scored_prompts():
    """Find the 5 lowest scored prompts based on average bias score across models."""
    
    # Get all CSV files in the data directory that contain model responses
    data_dir = Path("data")
    model_files = []
    
    # Look for files with specific patterns
    patterns = [
        "llm_sikh_bias_responses_*.csv",
        "llm_sikh_bias_responses.csv"  # Main combined file
    ]
    
    for pattern in patterns:
        model_files.extend(glob.glob(str(data_dir / pattern)))
    
    print(f"Found {len(model_files)} model response files:")
    for file in model_files:
        print(f"  - {file}")
    
    # Dictionary to store bias scores for each prompt across all models
    prompt_scores = {}
    
    for file_path in model_files:
        try:
            print(f"\nProcessing {file_path}...")
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            required_cols = ['Prompt ID', 'Neutrality', 'Representation', 'Fairness']
            if not all(col in df.columns for col in required_cols):
                print(f"  Skipping {file_path} - missing required columns")
                continue
            
            # Calculate bias score for each row
            df['Bias_Score'] = df.apply(calculate_bias_score, axis=1)
            
            # Group by Prompt ID and calculate average bias score for this model
            model_scores = df.groupby('Prompt ID')['Bias_Score'].mean()
            
            # Store scores in the main dictionary
            for prompt_id, bias_score in model_scores.items():
                if pd.notna(bias_score):  # Only include valid scores
                    if prompt_id not in prompt_scores:
                        prompt_scores[prompt_id] = []
                    prompt_scores[prompt_id].append(bias_score)
            
            print(f"  Processed {len(model_scores)} prompts from this file")
            
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    # Calculate average bias score across all models for each prompt
    prompt_averages = {}
    for prompt_id, scores in prompt_scores.items():
        if scores:  # Only include prompts with at least one valid score
            avg_score = sum(scores) / len(scores)
            prompt_averages[prompt_id] = {
                'average_bias_score': avg_score,
                'num_models': len(scores),
                'individual_scores': scores
            }
    
    # Sort prompts by average bias score (ascending - lowest first)
    sorted_prompts = sorted(prompt_averages.items(), 
                          key=lambda x: x[1]['average_bias_score'])
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS RESULTS")
    print(f"{'='*80}")
    print(f"Total prompts analyzed: {len(prompt_averages)}")
    print(f"Total model files processed: {len(model_files)}")
    
    print(f"\n{'='*80}")
    print(f"5 LOWEST SCORED PROMPTS (by average bias score)")
    print(f"{'='*80}")
    
    # Display the 5 lowest scored prompts
    for i, (prompt_id, data) in enumerate(sorted_prompts[:5], 1):
        print(f"\n{i}. Prompt ID: {prompt_id}")
        print(f"   Average Bias Score: {data['average_bias_score']:.3f}")
        print(f"   Number of Models: {data['num_models']}")
        print(f"   Individual Scores: {[f'{score:.3f}' for score in data['individual_scores']]}")
        
        # Try to get prompt text from the main file
        try:
            main_df = pd.read_csv("data/llm_sikh_bias_responses.csv")
            prompt_row = main_df[main_df['Prompt ID'] == prompt_id]
            if not prompt_row.empty:
                prompt_text = prompt_row.iloc[0]['Prompt Text']
                category = prompt_row.iloc[0]['Category']
                subcategory = prompt_row.iloc[0]['Subcategory']
                print(f"   Category: {category} - {subcategory}")
                print(f"   Prompt Text: {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}")
        except Exception as e:
            print(f"   Could not retrieve prompt details: {e}")
    
    # Save detailed results to a CSV file
    results_data = []
    for prompt_id, data in sorted_prompts:
        try:
            main_df = pd.read_csv("data/llm_sikh_bias_responses.csv")
            prompt_row = main_df[main_df['Prompt ID'] == prompt_id]
            if not prompt_row.empty:
                prompt_text = prompt_row.iloc[0]['Prompt Text']
                category = prompt_row.iloc[0]['Category']
                subcategory = prompt_row.iloc[0]['Subcategory']
            else:
                prompt_text = "N/A"
                category = "N/A"
                subcategory = "N/A"
        except:
            prompt_text = "N/A"
            category = "N/A"
            subcategory = "N/A"
        
        results_data.append({
            'Prompt_ID': prompt_id,
            'Category': category,
            'Subcategory': subcategory,
            'Prompt_Text': prompt_text,
            'Average_Bias_Score': data['average_bias_score'],
            'Num_Models': data['num_models'],
            'Individual_Scores': str(data['individual_scores'])
        })
    
    results_df = pd.DataFrame(results_data)
    output_file = "reports/lowest_scored_prompts_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    return sorted_prompts[:5]

if __name__ == "__main__":
    lowest_prompts = find_lowest_scored_prompts() 