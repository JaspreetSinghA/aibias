#!/usr/bin/env python3
"""
Script to fix the format of adjusted response files to match the correct structure.
Removes 'Bias Score' column and adds proper rating columns (Accuracy, Relevance, Fairness, Neutrality, Representation).
"""

import os
import pandas as pd
from pathlib import Path
import glob

def fix_response_file_format(file_path):
    """Fix the format of a single response file."""
    print(f"Processing: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check if the file has the wrong format (has 'Bias Score' column)
    if 'Bias Score (1-5)' in df.columns:
        print(f"  - Found 'Bias Score' column, fixing format...")
        
        # Remove the 'Bias Score (1-5)' column
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
        
        # Add any missing columns
        for col in correct_order:
            if col not in df.columns:
                df[col] = ''
        
        # Reorder columns
        df = df[correct_order]
        
        # Save the fixed file
        df.to_csv(file_path, index=False)
        print(f"  ✅ Fixed format: {file_path}")
        return True
    else:
        print(f"  - File already has correct format")
        return False

def main():
    """Fix all adjusted response files in the mitigation workflow directory."""
    
    adjusted_responses_dir = "data/mitigation_workflow/adjusted_responses"
    
    if not os.path.exists(adjusted_responses_dir):
        print(f"Directory not found: {adjusted_responses_dir}")
        return
    
    # Find all CSV files in the adjusted_responses directory
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