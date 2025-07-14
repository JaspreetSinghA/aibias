#!/usr/bin/env python3
"""
Debug script to check baseline and mitigated data matching
"""

import pandas as pd
from pathlib import Path

def debug_data_matching():
    # Load baseline data
    baseline_files = [
        'data/llm_sikh_bias_responses_Narveer_claude-3-haiku-20240307_adjusted_20250708_175346.csv',
        'data/llm_sikh_bias_responses_Noor_gpt-4.csv',
        'data/llm_sikh_bias_responses_Jaspreet_claude-3-haiku-20240307.csv',
        'data/llm_sikh_bias_responses_Harpreet_llama-3.3-70b-versatile.csv',
        'data/llm_sikh_bias_responses_Gurleen_gpt-4.csv',
        'data/llm_sikh_bias_responses_Anu_llama-3.3-70b-versatile.csv'
    ]
    
    baseline_data = []
    for file in baseline_files:
        if Path(file).exists():
            df = pd.read_csv(file)
            df['Strategy'] = 'Baseline'
            baseline_data.append(df)
    
    if baseline_data:
        baseline_df = pd.concat(baseline_data, ignore_index=True)
        print(f"Baseline data: {len(baseline_df)} rows")
        print(f"Baseline columns: {list(baseline_df.columns)}")
        print(f"Baseline models: {baseline_df['Model'].unique()}")
        print(f"Baseline prompt IDs: {len(baseline_df['Prompt ID'].unique())}")
        print(f"Sample baseline prompt IDs: {baseline_df['Prompt ID'].unique()[:10]}")
    else:
        print("No baseline files found")
        return
    
    # Load mitigated data
    strategy_dir = "data/mitigation_workflow/prompt_engineering_strategy/standardized_graded_responses"
    consolidated_file = Path(strategy_dir) / "all_standardized_graded_responses.csv"
    
    if consolidated_file.exists():
        mitigated_df = pd.read_csv(consolidated_file)
        print(f"\nMitigated data: {len(mitigated_df)} rows")
        print(f"Mitigated columns: {list(mitigated_df.columns)}")
        print(f"Mitigated models: {mitigated_df['Model'].unique()}")
        print(f"Mitigated strategies: {mitigated_df['Strategy'].unique()}")
        print(f"Mitigated prompt IDs: {len(mitigated_df['Prompt ID'].unique())}")
        print(f"Sample mitigated prompt IDs: {mitigated_df['Prompt ID'].unique()[:10]}")
    else:
        print(f"Mitigated file not found: {consolidated_file}")
        return
    
    # Check for common prompts
    baseline_prompts = set(zip(baseline_df['Prompt ID'], baseline_df['Model']))
    mitigated_prompts = set(zip(mitigated_df['Prompt ID'], mitigated_df['Model']))
    
    print(f"\nBaseline unique (Prompt ID, Model) pairs: {len(baseline_prompts)}")
    print(f"Mitigated unique (Prompt ID, Model) pairs: {len(mitigated_prompts)}")
    
    common_prompts = baseline_prompts.intersection(mitigated_prompts)
    print(f"Common (Prompt ID, Model) pairs: {len(common_prompts)}")
    
    if len(common_prompts) == 0:
        print("\nNo common prompts found. Let's investigate:")
        
        # Check if it's a model name mismatch
        print(f"\nBaseline model names: {sorted(baseline_df['Model'].unique())}")
        print(f"Mitigated model names: {sorted(mitigated_df['Model'].unique())}")
        
        # Check if it's a prompt ID mismatch
        baseline_prompt_ids = set(baseline_df['Prompt ID'].unique())
        mitigated_prompt_ids = set(mitigated_df['Prompt ID'].unique())
        
        print(f"\nBaseline prompt IDs: {sorted(baseline_prompt_ids)}")
        print(f"Mitigated prompt IDs: {sorted(mitigated_prompt_ids)}")
        
        # Check for any overlap in prompt IDs
        common_prompt_ids = baseline_prompt_ids.intersection(mitigated_prompt_ids)
        print(f"Common prompt IDs (ignoring model): {len(common_prompt_ids)}")
        if len(common_prompt_ids) > 0:
            print(f"Sample common prompt IDs: {list(common_prompt_ids)[:5]}")
            
            # Check what models are used for these common prompt IDs
            for prompt_id in list(common_prompt_ids)[:3]:
                baseline_models = baseline_df[baseline_df['Prompt ID'] == prompt_id]['Model'].unique()
                mitigated_models = mitigated_df[mitigated_df['Prompt ID'] == prompt_id]['Model'].unique()
                print(f"Prompt {prompt_id}:")
                print(f"  Baseline models: {baseline_models}")
                print(f"  Mitigated models: {mitigated_models}")

if __name__ == "__main__":
    debug_data_matching() 