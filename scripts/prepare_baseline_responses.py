#!/usr/bin/env python3
"""
Prepare baseline response files for bias diagnostic analysis

This script cleans the original response files by:
1. Removing rating columns (Accuracy, Relevence, Fairness, Neutrality, Representation)
2. Renaming the response column to 'response' (lowercase)
3. Adding a 'model' column to identify the source model
4. Saving cleaned files in the baseline_responses folder
"""

import pandas as pd
import os
from pathlib import Path

def clean_response_file(input_file, output_file, model_name):
    """
    Clean a response file and save it in the proper format for bias analysis
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        model_name (str): Name of the model (e.g., 'claude', 'llama', 'gpt')
    """
    print(f"Processing {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Identify the response column (it's the 5th column, index 4)
    response_col = df.columns[4]
    print(f"Found response column: {response_col}")
    
    # Create a new dataframe with only the essential columns
    cleaned_df = pd.DataFrame({
        'prompt_id': df['Prompt ID'],
        'prompt_text': df['Prompt Text'],
        'category': df['Category'],
        'subcategory': df['Subcategory'],
        'response': df[response_col],
        'model': model_name
    })
    
    # Remove rows where response is empty or NaN
    cleaned_df = cleaned_df.dropna(subset=['response'])
    cleaned_df = cleaned_df[cleaned_df['response'].str.strip() != '']
    
    # Save the cleaned file
    cleaned_df.to_csv(output_file, index=False)
    print(f"Saved {len(cleaned_df)} responses to {output_file}")
    
    return len(cleaned_df)

def main():
    """Main function to process all baseline response files"""
    
    # Define input files and their corresponding model names
    input_files = [
        ('Sikh Biases LLM - Arjun - LLM#2.csv', 'claude'),
        ('Sikh Biases LLM - Jugraj Paji - LLM#1.csv', 'llama'),
        ('Sikh Biases LLM - Tanvi - LLM#3.csv', 'gpt')
    ]
    
    # Create output directory
    output_dir = Path('data/mitigation_workflow/semantic_similarity_strategy/original_baseline_responses')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_responses = 0
    
    # Process each file
    for input_file, model_name in input_files:
        if os.path.exists(input_file):
            output_file = output_dir / f"baseline_{model_name}_responses.csv"
            count = clean_response_file(input_file, output_file, model_name)
            total_responses += count
        else:
            print(f"Warning: {input_file} not found")
    
    print(f"\nProcessing complete!")
    print(f"Total responses processed: {total_responses}")
    print(f"Files saved in: {output_dir}")
    
    # List the created files
    print("\nCreated files:")
    for file in output_dir.glob("*.csv"):
        print(f"  - {file}")

if __name__ == "__main__":
    main() 