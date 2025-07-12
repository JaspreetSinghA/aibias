#!/usr/bin/env python3
"""
Final robust script to fix the format of adjusted response files to match the correct structure.
Removes 'Bias Score' column and adds proper rating columns while preserving the actual responses.
"""

import os
import pandas as pd
import glob
import re

def get_model_response_column(filename, df):
    """Infer the correct LLM response column based on the filename and available columns."""
    # Lowercase columns for matching
    columns = [col.lower() for col in df.columns]
    # Try to infer model from filename
    fname = filename.lower()
    if 'gpt_4' in fname or 'gpt-4' in fname:
        for col in df.columns:
            if col.lower() == 'gpt-4-response':
                return col
    if 'claude' in fname:
        for col in df.columns:
            if 'claude' in col.lower() and col.lower().endswith('-response'):
                return col
    if 'llama' in fname:
        for col in df.columns:
            if 'llama' in col.lower() and col.lower().endswith('-response'):
                return col
    # Fallback: pick the first column ending with '-response' that is not 'Response'
    for col in df.columns:
        if col.lower().endswith('-response') and col.lower() != 'response':
            return col
    return None

def fix_response_file_format(file_path):
    """Fix the format of a single response file while preserving responses."""
    print(f"Processing: {file_path}")
    df = pd.read_csv(file_path)
    if 'Bias Score (1-5)' in df.columns:
        print(f"  - Found 'Bias Score' column, fixing format...")
        # Find the correct LLM response column
        model_response_col = get_model_response_column(file_path, df)
        if model_response_col:
            print(f"  - Using response column: {model_response_col}")
            df['Response'] = df[model_response_col]
            # Remove the model-specific response column and Bias Score column
            drop_cols = [model_response_col, 'Bias Score (1-5)']
            # Only drop if present
            drop_cols = [c for c in drop_cols if c in df.columns]
            df = df.drop(columns=drop_cols)
        else:
            print(f"  - Warning: No model response column found, only removing Bias Score column.")
            df = df.drop(columns=['Bias Score (1-5)'])
        # Add the proper rating columns if they don't exist
        rating_columns = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
        for col in rating_columns:
            if col not in df.columns:
                df[col] = ''
        # Reorder columns to match the correct format
        correct_order = [
            'Prompt ID', 'Prompt Text', 'Category', 'Subcategory', 'Model',
            'Response', 'Accuracy', 'Relevance', 'Fairness', 'Neutrality',
            'Representation', 'Comments'
        ]
        for col in correct_order:
            if col not in df.columns:
                df[col] = ''
        df = df[correct_order]
        df.to_csv(file_path, index=False)
        print(f"  ✅ Fixed format: {file_path}")
        return True
    else:
        print(f"  - File already has correct format")
        return False

def main():
    adjusted_responses_dir = "data/mitigation_workflow/adjusted_responses"
    if not os.path.exists(adjusted_responses_dir):
        print(f"Directory not found: {adjusted_responses_dir}")
        return
    csv_files = glob.glob(os.path.join(adjusted_responses_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found in adjusted_responses directory")
        return
    print(f"Found {len(csv_files)} CSV files to process")
    print("=" * 60)
    fixed_count = 0
    for file_path in csv_files:
        if fix_response_file_format(file_path):
            fixed_count += 1
    print("=" * 60)
    print(f"Processing complete!")
    print(f"Fixed {fixed_count} out of {len(csv_files)} files")
    if fixed_count > 0:
        print("\nFixed files now have the correct format with columns:")
        print("Prompt ID, Prompt Text, Category, Subcategory, Model, Response, Accuracy, Relevance, Fairness, Neutrality, Representation, Comments")
        print("\nYou can now use these files for grading with aidata.py")

if __name__ == "__main__":
    main() 