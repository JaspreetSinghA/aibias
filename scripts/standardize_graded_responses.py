#!/usr/bin/env python3
"""
Standardize Graded Response Files
=================================

This script standardizes the format of graded response files by:
1. Converting star ratings (★☆☆☆☆) to numeric values (1-5)
2. Ensuring consistent column structure
3. Adding missing model information
4. Standardizing the format across all files
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime

def convert_star_rating_to_numeric(rating):
    """Convert star ratings to numeric values."""
    if pd.isna(rating) or rating == '':
        return np.nan
    
    rating_str = str(rating).strip()
    
    # If already numeric, return as is
    if rating_str.isdigit() and 1 <= int(rating_str) <= 5:
        return int(rating_str)
    
    # Convert star ratings to numeric
    if '★' in rating_str:
        # Count filled stars
        filled_stars = rating_str.count('★')
        return filled_stars
    
    # Handle other formats
    if rating_str.lower() in ['one', '1', 'one star', '1 star']:
        return 1
    elif rating_str.lower() in ['two', '2', 'two stars', '2 stars']:
        return 2
    elif rating_str.lower() in ['three', '3', 'three stars', '3 stars']:
        return 3
    elif rating_str.lower() in ['four', '4', 'four stars', '4 stars']:
        return 4
    elif rating_str.lower() in ['five', '5', 'five stars', '5 stars']:
        return 5
    
    return np.nan

def extract_model_from_filename(filename):
    """Extract model name from filename."""
    filename_lower = filename.lower()
    
    if 'claude' in filename_lower:
        return 'CLAUDE'
    elif 'gpt' in filename_lower:
        return 'GPT'
    elif 'llama' in filename_lower:
        return 'LLAMA'
    else:
        return 'UNKNOWN'

def extract_strategy_from_filename(filename):
    """Extract strategy name from filename."""
    filename_lower = filename.lower()
    
    if 'contextual' in filename_lower:
        return 'contextual'
    elif 'instructional' in filename_lower:
        return 'instructional'
    elif 'retrieval' in filename_lower:
        return 'retrieval_based'
    else:
        return 'unknown'

def standardize_graded_response_file(file_path):
    """Standardize a single graded response file."""
    print(f"Processing: {file_path.name}")
    
    # Read the file
    df = pd.read_csv(file_path)
    
    # Extract model and strategy from filename
    model = extract_model_from_filename(file_path.name)
    strategy = extract_strategy_from_filename(file_path.name)
    
    # Add model and strategy columns if missing
    if 'Model' not in df.columns or df['Model'].isna().all():
        df['Model'] = model
    
    if 'Strategy' not in df.columns:
        df['Strategy'] = strategy
    
    # Convert star ratings to numeric for all rating columns
    rating_columns = ['Accuracy (1-5)', 'Relevance (1-5)', 'Fairness (1-5)', 
                     'Neutrality (1-5)', 'Representation (1-5)']
    
    for col in rating_columns:
        if col in df.columns:
            df[col] = df[col].apply(convert_star_rating_to_numeric)
    
    # Calculate average score
    numeric_columns = [col for col in rating_columns if col in df.columns]
    if numeric_columns:
        df['Average_Score'] = df[numeric_columns].mean(axis=1)
    
    # Ensure all required columns exist
    required_columns = ['Prompt ID', 'Prompt Text', 'Category', 'Subcategory', 
                       'Model', 'Strategy', 'Response', 'Accuracy (1-5)', 
                       'Relevance (1-5)', 'Fairness (1-5)', 'Neutrality (1-5)', 
                       'Representation (1-5)', 'Comments', 'Average_Score']
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    # Reorder columns
    df = df[required_columns]
    
    return df

def main():
    """Main execution function."""
    print("Starting Graded Response Standardization...")
    
    # Define input and output directories
    input_dir = Path("data/mitigation_workflow/prompt_engineering_strategy/adjusted_gradedresponses")
    output_dir = Path("data/mitigation_workflow/prompt_engineering_strategy/standardized_graded_responses")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all CSV files in the input directory
    csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in the input directory.")
        return
    
    all_data = []
    
    for file_path in csv_files:
        try:
            standardized_df = standardize_graded_response_file(file_path)
            
            # Save individual standardized file
            output_filename = f"standardized_{file_path.name}"
            output_path = output_dir / output_filename
            standardized_df.to_csv(output_path, index=False)
            
            # Add to combined dataset
            all_data.append(standardized_df)
            
            print(f"✓ Standardized: {file_path.name}")
            
        except Exception as e:
            print(f"✗ Error processing {file_path.name}: {str(e)}")
    
    # Create combined dataset
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save combined dataset
        combined_path = output_dir / "all_standardized_graded_responses.csv"
        combined_df.to_csv(combined_path, index=False)
        
        # Generate summary statistics
        summary_stats = {
            'Total_Responses': len(combined_df),
            'Models': combined_df['Model'].value_counts().to_dict(),
            'Strategies': combined_df['Strategy'].value_counts().to_dict(),
            'Categories': combined_df['Category'].value_counts().to_dict(),
            'Average_Scores_by_Model': combined_df.groupby('Model')['Average_Score'].mean().to_dict(),
            'Average_Scores_by_Strategy': combined_df.groupby('Strategy')['Average_Score'].mean().to_dict(),
            'Average_Scores_by_Category': combined_df.groupby('Category')['Average_Score'].mean().to_dict()
        }
        
        # Save summary statistics
        summary_path = output_dir / "graded_responses_summary.csv"
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nStandardization complete!")
        print(f"Individual files saved to: {output_dir}")
        print(f"Combined dataset: {combined_path}")
        print(f"Summary statistics: {summary_path}")
        print(f"\nSummary:")
        print(f"- Total responses: {len(combined_df)}")
        print(f"- Models: {list(combined_df['Model'].unique())}")
        print(f"- Strategies: {list(combined_df['Strategy'].unique())}")
        print(f"- Overall average score: {combined_df['Average_Score'].mean():.2f}")
        
        return combined_df
    
    else:
        print("No files were successfully processed.")

if __name__ == "__main__":
    main() 