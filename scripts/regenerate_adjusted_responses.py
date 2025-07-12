#!/usr/bin/env python3
"""
Script to regenerate all adjusted responses by running aidata.py on the adjusted prompt files.
This ensures we get fresh, complete responses from all models.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def run_aidata_on_file(prompt_file, model_name):
    """Run aidata.py on a specific prompt file with a specific model."""
    print(f"Running aidata.py on {prompt_file} with model {model_name}")
    
    # Set the model in the config
    config_file = "config/config.py"
    
    # Read current config
    with open(config_file, 'r') as f:
        config_content = f.read()
    
    # Update the model
    if 'MODEL_NAME = ' in config_content:
        # Replace the model name
        lines = config_content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('MODEL_NAME = '):
                lines[i] = f'MODEL_NAME = "{model_name}"'
                break
        config_content = '\n'.join(lines)
    
    # Write updated config
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    # Run aidata.py
    try:
        result = subprocess.run([sys.executable, "scripts/aidata.py", prompt_file], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ Successfully generated responses for {model_name}")
            return True
        else:
            print(f"❌ Error generating responses for {model_name}:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout generating responses for {model_name}")
        return False
    except Exception as e:
        print(f"❌ Exception generating responses for {model_name}: {e}")
        return False

def main():
    """Regenerate all adjusted responses."""
    
    # Define the models and their config names
    models = [
        ("gpt-4", "gpt-4"),
        ("claude-3-haiku-20240307", "claude-3-haiku-20240307"),
        ("llama-3.3-70b-versatile", "llama-3.3-70b-versatile")
    ]
    
    # Define the adjusted prompt files
    prompt_files = [
        "data/mitigation_workflow/adjusted_prompts/instructional_strategy_prompts.csv",
        "data/mitigation_workflow/adjusted_prompts/contextual_strategy_prompts.csv", 
        "data/mitigation_workflow/adjusted_prompts/retrieval_based_strategy_prompts.csv"
    ]
    
    # Check if prompt files exist
    for prompt_file in prompt_files:
        if not os.path.exists(prompt_file):
            print(f"❌ Prompt file not found: {prompt_file}")
            return
    
    print("Starting regeneration of adjusted responses...")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    output_dir = "data/mitigation_workflow/adjusted_responses"
    os.makedirs(output_dir, exist_ok=True)
    
    # Track results
    successful_runs = 0
    total_runs = len(models) * len(prompt_files)
    
    # Run aidata.py for each model and prompt file combination
    for model_display, model_config in models:
        for prompt_file in prompt_files:
            print(f"\n--- Processing {os.path.basename(prompt_file)} with {model_display} ---")
            
            if run_aidata_on_file(prompt_file, model_config):
                successful_runs += 1
            
            # Small delay between runs
            time.sleep(2)
    
    print("\n" + "=" * 60)
    print(f"Regeneration complete!")
    print(f"Successful runs: {successful_runs}/{total_runs}")
    
    if successful_runs == total_runs:
        print("✅ All response files generated successfully!")
        print("\nGenerated files should be in:")
        print(f"  {output_dir}/")
        print("\nYou can now run the format fixing script to ensure proper structure.")
    else:
        print("⚠️  Some files failed to generate. Check the output above for errors.")

if __name__ == "__main__":
    main() 