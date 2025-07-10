#!/usr/bin/env python3
"""
Script to run aidata.py on each mitigation strategy CSV file
and save responses to the adjusted_responses folder.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

def run_aidata_on_strategy(strategy_file, output_dir):
    """Run aidata.py on a specific strategy file and save results."""
    
    strategy_name = Path(strategy_file).stem.replace('_strategy_prompts', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"Processing strategy: {strategy_name}")
    print(f"Input file: {strategy_file}")
    print(f"{'='*60}")
    
    # Create a temporary copy of the strategy file in the data directory
    # (aidata.py expects to read from data/llm_sikh_bias_prompts.csv)
    temp_prompts_file = "data/llm_sikh_bias_prompts.csv"
    temp_responses_file = "llm_sikh_bias_responses.csv"
    
    # Backup original files if they exist
    original_prompts_backup = None
    original_responses_backup = None
    
    if os.path.exists(temp_prompts_file):
        original_prompts_backup = f"{temp_prompts_file}.backup"
        shutil.copy2(temp_prompts_file, original_prompts_backup)
        print(f"Backed up original prompts file to: {original_prompts_backup}")
    
    if os.path.exists(temp_responses_file):
        original_responses_backup = f"{temp_responses_file}.backup"
        shutil.copy2(temp_responses_file, original_responses_backup)
        print(f"Backed up original responses file to: {original_responses_backup}")
    
    try:
        # Copy the strategy file to the expected location
        shutil.copy2(strategy_file, temp_prompts_file)
        print(f"Copied {strategy_file} to {temp_prompts_file}")
        
        # Run aidata.py
        print(f"Running aidata.py...")
        result = subprocess.run([sys.executable, "scripts/aidata.py"], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ aidata.py completed successfully!")
            
            # Check if responses file was created
            if os.path.exists(temp_responses_file):
                # Create output filename
                output_filename = f"{strategy_name}_responses_{timestamp}.csv"
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
        print(f"❌ Error processing {strategy_name}: {e}")
        return None
        
    finally:
        # Restore original files
        if original_prompts_backup and os.path.exists(original_prompts_backup):
            shutil.move(original_prompts_backup, temp_prompts_file)
            print(f"Restored original prompts file")
        
        if original_responses_backup and os.path.exists(original_responses_backup):
            shutil.move(original_responses_backup, temp_responses_file)
            print(f"Restored original responses file")

def main():
    """Main function to process all three strategy files."""
    
    # Define paths
    adjusted_prompts_dir = "data/mitigation_workflow/adjusted_prompts"
    adjusted_responses_dir = "data/mitigation_workflow/adjusted_responses"
    
    # Strategy files to process
    strategy_files = [
        os.path.join(adjusted_prompts_dir, "instructional_strategy_prompts.csv"),
        os.path.join(adjusted_prompts_dir, "contextual_strategy_prompts.csv"),
        os.path.join(adjusted_prompts_dir, "retrieval_based_strategy_prompts.csv")
    ]
    
    # Ensure output directory exists
    os.makedirs(adjusted_responses_dir, exist_ok=True)
    
    print("🚀 Starting mitigation strategy processing...")
    print(f"Input directory: {adjusted_prompts_dir}")
    print(f"Output directory: {adjusted_responses_dir}")
    
    results = {}
    
    # Process each strategy file
    for strategy_file in strategy_files:
        if os.path.exists(strategy_file):
            output_file = run_aidata_on_strategy(strategy_file, adjusted_responses_dir)
            if output_file:
                strategy_name = Path(strategy_file).stem.replace('_strategy_prompts', '')
                results[strategy_name] = output_file
        else:
            print(f"❌ Strategy file not found: {strategy_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    if results:
        print("✅ Successfully processed strategies:")
        for strategy, output_file in results.items():
            print(f"  {strategy}: {output_file}")
    else:
        print("❌ No strategies were processed successfully")
    
    print(f"\nOutput files are in: {adjusted_responses_dir}")
    print("You can now use these response files with your grading workflow.")

if __name__ == "__main__":
    main() 