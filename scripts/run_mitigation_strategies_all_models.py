#!/usr/bin/env python3
"""
Script to run aidata.py on each mitigation strategy CSV file
with all three models (GPT-4, Claude, and Llama) and save responses.
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime

def update_config_for_model(model_name):
    """Temporarily update the config to use a specific model."""
    config_file = "config/config.py"
    
    # Read the current config
    with open(config_file, 'r') as f:
        config_content = f.read()
    
    # Update the models_to_run list
    if model_name == 'gpt-4':
        new_models_line = "    'models_to_run': [\n        'gpt-4',\n    ],"
    elif model_name == 'claude-3-haiku-20240307':
        new_models_line = "    'models_to_run': [\n        'claude-3-haiku-20240307',\n    ],"
    elif model_name == 'llama-3.3-70b-versatile':
        new_models_line = "    'models_to_run': [\n        'llama-3.3-70b-versatile',\n    ],"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Replace the models_to_run section
    import re
    pattern = r"    'models_to_run': \[\s*\n\s*'[^']+',\s*\n\s*\],"
    config_content = re.sub(pattern, new_models_line, config_content)
    
    # Write back to config
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"Updated config to use {model_name}")

def run_aidata_on_strategy_with_model(strategy_file, model_name, output_dir):
    """Run aidata.py on a specific strategy file with a specific model and save results."""
    
    strategy_name = Path(strategy_file).stem.replace('_strategy_prompts', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"Processing strategy: {strategy_name} with model: {model_name}")
    print(f"Input file: {strategy_file}")
    print(f"{'='*60}")
    
    # Create a temporary copy of the strategy file in the data directory
    temp_prompts_file = "data/llm_sikh_bias_prompts.csv"
    temp_responses_file = "llm_sikh_bias_responses.csv"
    
    # Backup original files if they exist
    original_prompts_backup = None
    original_responses_backup = None
    original_config_backup = None
    
    if os.path.exists(temp_prompts_file):
        original_prompts_backup = f"{temp_prompts_file}.backup"
        shutil.copy2(temp_prompts_file, original_prompts_backup)
        print(f"Backed up original prompts file to: {original_prompts_backup}")
    
    if os.path.exists(temp_responses_file):
        original_responses_backup = f"{temp_responses_file}.backup"
        shutil.copy2(temp_responses_file, original_responses_backup)
        print(f"Backed up original responses file to: {original_responses_backup}")
    
    # Backup and update config
    config_file = "config/config.py"
    original_config_backup = f"{config_file}.backup"
    shutil.copy2(config_file, original_config_backup)
    print(f"Backed up original config file to: {original_config_backup}")
    
    try:
        # Update config for the specific model
        update_config_for_model(model_name)
        
        # Copy the strategy file to the expected location
        shutil.copy2(strategy_file, temp_prompts_file)
        print(f"Copied {strategy_file} to {temp_prompts_file}")
        
        # Run aidata.py
        print(f"Running aidata.py with {model_name}...")
        result = subprocess.run([sys.executable, "scripts/aidata.py"], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ aidata.py completed successfully!")
            
            # Check if responses file was created
            if os.path.exists(temp_responses_file):
                # Create output filename
                output_filename = f"{strategy_name}_{model_name.replace('-', '_')}_responses_{timestamp}.csv"
                output_path = os.path.join(output_dir, output_filename)
                
                # Move the responses file to the output directory
                shutil.move(temp_responses_file, output_path)
                print(f"✅ Responses saved to: {output_path}")
                
                return output_path
            else:
                print("❌ No responses file was created by aidata.py")
                return None
        else:
            print(f"❌ aidata.py failed with return code: {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error processing {strategy_name} with {model_name}: {e}")
        return None
        
    finally:
        # Restore original files
        if original_prompts_backup and os.path.exists(original_prompts_backup):
            shutil.move(original_prompts_backup, temp_prompts_file)
            print(f"Restored original prompts file")
        
        if original_responses_backup and os.path.exists(original_responses_backup):
            shutil.move(original_responses_backup, temp_responses_file)
            print(f"Restored original responses file")
        
        if original_config_backup and os.path.exists(original_config_backup):
            shutil.move(original_config_backup, config_file)
            print(f"Restored original config file")

def main():
    """Main function to process all three strategy files with all three models."""
    
    # Define paths
    adjusted_prompts_dir = "data/mitigation_workflow/adjusted_prompts"
    adjusted_responses_dir = "data/mitigation_workflow/adjusted_responses"
    
    # Strategy files to process
    strategy_files = [
        os.path.join(adjusted_prompts_dir, "instructional_strategy_prompts.csv"),
        os.path.join(adjusted_prompts_dir, "contextual_strategy_prompts.csv"),
        os.path.join(adjusted_prompts_dir, "retrieval_based_strategy_prompts.csv")
    ]
    
    # Models to run
    models = [
        'gpt-4',
        'claude-3-haiku-20240307',
        'llama-3.3-70b-versatile'
    ]
    
    # Ensure output directory exists
    os.makedirs(adjusted_responses_dir, exist_ok=True)
    
    print("🚀 Starting mitigation strategy processing with all models...")
    print(f"Input directory: {adjusted_prompts_dir}")
    print(f"Output directory: {adjusted_responses_dir}")
    print(f"Models: {', '.join(models)}")
    
    results = {}
    
    # Process each strategy file with each model
    for strategy_file in strategy_files:
        if os.path.exists(strategy_file):
            strategy_name = Path(strategy_file).stem.replace('_strategy_prompts', '')
            results[strategy_name] = {}
            
            for model in models:
                output_file = run_aidata_on_strategy_with_model(strategy_file, model, adjusted_responses_dir)
                if output_file:
                    results[strategy_name][model] = output_file
                else:
                    print(f"❌ Failed to process {strategy_name} with {model}")
        else:
            print(f"❌ Strategy file not found: {strategy_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    if results:
        print("✅ Successfully processed strategies:")
        for strategy, model_results in results.items():
            print(f"\n  {strategy}:")
            for model, output_file in model_results.items():
                print(f"    {model}: {output_file}")
    else:
        print("❌ No strategies were processed successfully")
    
    print(f"\nOutput files are in: {adjusted_responses_dir}")
    print("You can now use these response files with your grading workflow.")

if __name__ == "__main__":
    main() 