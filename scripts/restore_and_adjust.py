#!/usr/bin/env python3
"""
Restore Original Model Files and Apply Correct Adjustments
This script restores the original model files from the archive and then applies
the correct adjustments, preserving legitimate 5s while only adjusting inflated scores.
"""

import pandas as pd
import os
import shutil
from datetime import datetime

def restore_original_files():
    """Restore original model files from archive"""
    
    print("=" * 80)
    print("RESTORING ORIGINAL MODEL FILES")
    print("=" * 80)
    
    # File mappings for restoration
    restore_mappings = [
        {
            'source': '../visualizations/archive/Sikh Biases LLM - Narveer - LLM#2.csv',
            'target': '../data/llm_sikh_bias_responses_Narveer_claude-3-haiku-20240307.csv',
            'rater_name': 'Narveer'
        },
        {
            'source': '../visualizations/archive/Sikh Biases LLM - Jaspreet - LLM#2.csv',
            'target': '../data/llm_sikh_bias_responses_Jaspreet_claude-3-haiku-20240307.csv',
            'rater_name': 'Jaspreet'
        }
    ]
    
    for mapping in restore_mappings:
        print(f"Restoring {mapping['rater_name']} file...")
        
        if os.path.exists(mapping['source']):
            # Read the archive file
            df = pd.read_csv(mapping['source'])
            
            # Add Model column if it doesn't exist
            if 'Model' not in df.columns:
                df['Model'] = 'claude-3-haiku-20240307'
            
            # Reorder columns to match expected format
            expected_columns = ['Prompt ID', 'Prompt Text', 'Category', 'Subcategory', 'Model', 'Response', 'Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Comments']
            
            # Rename CH column to Response if it exists
            if 'CH' in df.columns:
                df = df.rename(columns={'CH': 'Response'})
            
            # Rename Relevence to Relevance if it exists
            if 'Relevence' in df.columns:
                df = df.rename(columns={'Relevence': 'Relevance'})
            
            # Ensure all expected columns exist
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = ''
            
            # Reorder columns
            df = df[expected_columns]
            
            # Save as the model file
            df.to_csv(mapping['target'], index=False)
            print(f"  Restored: {mapping['target']}")
        else:
            print(f"  ERROR: Source file not found: {mapping['source']}")
    
    print("Restoration completed!")

def apply_correct_adjustments():
    """Apply adjustments only where model score > rater score"""
    
    print("\n" + "=" * 80)
    print("APPLYING CORRECT ADJUSTMENTS")
    print("=" * 80)
    
    # File mappings for adjustment
    adjust_mappings = [
        {
            'rater_file': '../visualizations/archive/Sikh Biases LLM - Narveer - LLM#2.csv',
            'model_file': '../data/llm_sikh_bias_responses_Narveer_claude-3-haiku-20240307.csv',
            'rater_name': 'Narveer'
        },
        {
            'rater_file': '../visualizations/archive/Sikh Biases LLM - Jaspreet - LLM#2.csv',
            'model_file': '../data/llm_sikh_bias_responses_Jaspreet_claude-3-haiku-20240307.csv',
            'rater_name': 'Jaspreet'
        }
    ]
    
    total_adjustments = 0
    
    for mapping in adjust_mappings:
        print(f"\nProcessing: {mapping['rater_name']}")
        print("-" * 60)
        
        # Load data
        rater_df = pd.read_csv(mapping['rater_file'])
        model_df = pd.read_csv(mapping['model_file'])
        
        # Convert score columns to numeric
        score_columns = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
        for col in score_columns:
            if col in rater_df.columns:
                rater_df[col] = pd.to_numeric(rater_df[col], errors='coerce')
            if col in model_df.columns:
                model_df[col] = pd.to_numeric(model_df[col], errors='coerce')
        
        # Apply adjustments
        adjustments_made = 0
        
        for idx, model_row in model_df.iterrows():
            prompt_id = model_row['Prompt ID']
            
            # Find corresponding rater row
            rater_row = rater_df[rater_df['Prompt ID'] == prompt_id]
            
            if len(rater_row) > 0:
                rater_row = rater_row.iloc[0]
                
                # Check each score column
                for col in score_columns:
                    if col in model_row and col in rater_row:
                        model_score = model_row[col]
                        rater_score = rater_row[col]
                        
                        # Skip if either score is missing
                        if pd.isna(model_score) or pd.isna(rater_score):
                            continue
                        
                        # Apply adjustment if model score > rater score
                        if model_score > rater_score:
                            model_df.loc[idx, col] = rater_score
                            adjustments_made += 1
        
        # Save adjusted file
        model_df.to_csv(mapping['model_file'], index=False)
        
        print(f"Applied {adjustments_made} score adjustments")
        total_adjustments += adjustments_made
    
    print(f"\nTotal adjustments applied: {total_adjustments}")
    print("Adjustments completed!")

def main():
    """Main function to restore and adjust"""
    
    print("=" * 80)
    print("RESTORE AND ADJUST MODEL FILES")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Restore original files
    restore_original_files()
    
    # Step 2: Apply correct adjustments
    apply_correct_adjustments()
    
    print("\n" + "=" * 80)
    print("PROCESS COMPLETED")
    print("=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Model files have been restored and correctly adjusted.")

if __name__ == "__main__":
    main() 