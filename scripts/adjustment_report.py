#!/usr/bin/env python3
"""
Adjustment Report Generator for Sikh Bias Analysis
Compares original rater files with model files to identify:
1. Score adjustments needed (when model score > rater score)
2. Outlier rows for pruning (all 5s or excessive 4s/5s)
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

def identify_adjustments(rater_df, model_df, rater_name, model_name):
    """Identify score adjustments and outliers"""
    adjustments = []
    outliers = []
    
    # Score columns to compare
    score_columns = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    
    # Merge on Prompt ID
    merged = pd.merge(rater_df, model_df, on='Prompt ID', suffixes=('_rater', '_model'))
    
    for _, row in merged.iterrows():
        prompt_id = row['Prompt ID']
        adjustments_for_prompt = []
        outlier_flags = []
        
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
                
                # Check if adjustment needed (model score > rater score)
                if model_score > rater_score:
                    adjustments_for_prompt.append({
                        'category': col,
                        'rater_score': rater_score,
                        'model_score': model_score,
                        'adjustment': f"{model_score} → {rater_score}"
                    })
        
        # Check for outliers (all 5s or excessive 4s/5s)
        model_scores = []
        for col in score_columns:
            model_col = f"{col}_model"
            if model_col in row and not pd.isna(row[model_col]):
                model_scores.append(row[model_col])
        
        if model_scores:
            high_scores = sum(1 for score in model_scores if score >= 4)
            if len(model_scores) == 5 and all(score == 5 for score in model_scores):
                outlier_flags.append("ALL_5S")
            elif high_scores >= 4:
                outlier_flags.append(f"EXCESSIVE_HIGH_SCORES ({high_scores}/5 scores ≥ 4)")
        
        # Record adjustments and outliers
        if adjustments_for_prompt:
            adjustments.append({
                'prompt_id': prompt_id,
                'rater': rater_name,
                'model': model_name,
                'adjustments': adjustments_for_prompt
            })
        
        if outlier_flags:
            outliers.append({
                'prompt_id': prompt_id,
                'rater': rater_name,
                'model': model_name,
                'flags': outlier_flags,
                'scores': model_scores
            })
    
    return adjustments, outliers

def generate_report():
    """Generate comprehensive adjustment report"""
    
    # File mappings
    file_mappings = [
        {
            'rater_file': '../visualizations/archive/Sikh Biases LLM - Narveer - LLM#2.csv',
            'model_file': '../data/llm_sikh_bias_responses_Narveer_claude-3-haiku-20240307.csv',
            'rater_name': 'Narveer',
            'model_name': 'claude-3-haiku-20240307'
        },
        {
            'rater_file': '../visualizations/archive/Sikh Biases LLM - Jaspreet - LLM#2.csv',
            'model_file': '../data/llm_sikh_bias_responses_Jaspreet_claude-3-haiku-20240307.csv',
            'rater_name': 'Jaspreet',
            'model_name': 'claude-3-haiku-20240307'
        }
    ]
    
    all_adjustments = []
    all_outliers = []
    
    print("=" * 80)
    print("SIKH BIAS ANALYSIS - ADJUSTMENT REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    for mapping in file_mappings:
        print(f"Processing: {mapping['rater_name']} → {mapping['model_name']}")
        print("-" * 60)
        
        # Load data
        rater_df = load_and_clean_data(mapping['rater_file'])
        model_df = load_and_clean_data(mapping['model_file'])
        
        if rater_df is None or model_df is None:
            print(f"ERROR: Could not load files for {mapping['rater_name']}")
            continue
        
        # Identify adjustments and outliers
        adjustments, outliers = identify_adjustments(
            rater_df, model_df, 
            mapping['rater_name'], mapping['model_name']
        )
        
        all_adjustments.extend(adjustments)
        all_outliers.extend(outliers)
        
        print(f"Found {len(adjustments)} prompts needing adjustments")
        print(f"Found {len(outliers)} outlier prompts")
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total adjustments needed: {len(all_adjustments)}")
    print(f"Total outliers identified: {len(all_outliers)}")
    print()
    
    # Detailed adjustments
    if all_adjustments:
        print("DETAILED ADJUSTMENTS")
        print("=" * 80)
        for adj in all_adjustments:
            print(f"\nPrompt ID: {adj['prompt_id']}")
            print(f"Rater: {adj['rater']} → Model: {adj['model']}")
            for change in adj['adjustments']:
                print(f"  {change['category']}: {change['adjustment']}")
    
    # Detailed outliers
    if all_outliers:
        print("\n" + "=" * 80)
        print("OUTLIERS FOR PRUNING")
        print("=" * 80)
        for outlier in all_outliers:
            print(f"\nPrompt ID: {outlier['prompt_id']}")
            print(f"Rater: {outlier['rater']} → Model: {outlier['model']}")
            print(f"Flags: {', '.join(outlier['flags'])}")
            print(f"Scores: {outlier['scores']}")
    
    # Save detailed report to file
    report_filename = f"../reports/adjustment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SIKH BIAS ANALYSIS - ADJUSTMENT REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total adjustments needed: {len(all_adjustments)}\n")
        f.write(f"Total outliers identified: {len(all_outliers)}\n\n")
        
        if all_adjustments:
            f.write("DETAILED ADJUSTMENTS\n")
            f.write("=" * 80 + "\n")
            for adj in all_adjustments:
                f.write(f"\nPrompt ID: {adj['prompt_id']}\n")
                f.write(f"Rater: {adj['rater']} → Model: {adj['model']}\n")
                for change in adj['adjustments']:
                    f.write(f"  {change['category']}: {change['adjustment']}\n")
        
        if all_outliers:
            f.write("\n" + "=" * 80 + "\n")
            f.write("OUTLIERS FOR PRUNING\n")
            f.write("=" * 80 + "\n")
            for outlier in all_outliers:
                f.write(f"\nPrompt ID: {outlier['prompt_id']}\n")
                f.write(f"Rater: {outlier['rater']} → Model: {outlier['model']}\n")
                f.write(f"Flags: {', '.join(outlier['flags'])}\n")
                f.write(f"Scores: {outlier['scores']}\n")
    
    print(f"\nDetailed report saved to: {report_filename}")
    
    return all_adjustments, all_outliers

if __name__ == "__main__":
    generate_report() 