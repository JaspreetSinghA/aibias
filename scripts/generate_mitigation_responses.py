#!/usr/bin/env python3
"""
Generate responses for mitigation strategies using multiple LLM models
Creates 9 CSV files (3 strategies x 3 models) in the adjusted_responses folder
"""

import os
import sys
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

from config.llm_clients import LLMManager
from config.config import MODEL_CONFIGS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_prompt_file(file_path):
    """Load a prompt CSV file and return the DataFrame"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} prompts from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def generate_responses_for_model(prompts_df, model_name, llm_manager, output_file):
    """Generate responses for a specific model and save to CSV"""
    logger.info(f"Generating responses for {model_name}...")
    
    # Create a copy of the dataframe
    result_df = prompts_df.copy()
    
    # Generate responses for each prompt
    responses = []
    for idx, row in result_df.iterrows():
        prompt_text = row['Prompt Text']
        logger.info(f"Processing prompt {idx + 1}/{len(result_df)}: {row['Prompt ID']}")
        
        try:
            response = llm_manager.query_model(prompt_text, model_name)
            responses.append(response)
            logger.info(f"Generated response for {row['Prompt ID']}")
        except Exception as e:
            logger.error(f"Error generating response for {row['Prompt ID']}: {e}")
            responses.append(f"Error: {e}")
    
    # Add responses to the dataframe
    result_df['Response'] = responses
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    logger.info(f"Saved responses to {output_file}")
    
    return result_df

def main():
    """Main function to generate responses for all strategies and models"""
    
    # Define the models to use
    models = {
        'gpt-4': 'GPT-4',
        'llama-3.3-70b-versatile': 'Llama-3.3-70B',
        'claude-3-haiku-20240307': 'Claude-3-Haiku'
    }
    
    # Define the strategy files
    strategies = {
        'instructional': 'data/mitigation_workflow/prompt_engineering_strategy/adjusted_prompts/instructional_strategy_prompts.csv',
        'contextual': 'data/mitigation_workflow/prompt_engineering_strategy/adjusted_prompts/contextual_strategy_prompts.csv',
        'retrieval_based': 'data/mitigation_workflow/prompt_engineering_strategy/adjusted_prompts/retrieval_based_strategy_prompts.csv'
    }
    
    # Create output directory
    output_dir = Path('data/mitigation_workflow/prompt_engineering_strategy/adjusted_responses')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize LLM manager
    logger.info("Initializing LLM manager...")
    llm_manager = LLMManager()
    
    # Check available models
    available_models = llm_manager.get_available_models()
    logger.info(f"Available models: {list(available_models.keys())}")
    
    # Generate responses for each strategy and model combination
    for strategy_name, prompt_file in strategies.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing strategy: {strategy_name}")
        logger.info(f"{'='*50}")
        
        # Load prompts
        prompts_df = load_prompt_file(prompt_file)
        if prompts_df is None:
            logger.error(f"Skipping {strategy_name} due to loading error")
            continue
        
        for model_name, model_display_name in models.items():
            if model_name not in available_models:
                logger.warning(f"Model {model_name} not available, skipping...")
                continue
            
            logger.info(f"\n--- Generating responses for {model_display_name} ---")
            
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{strategy_name}_{model_display_name.lower().replace('-', '_')}_{timestamp}.csv"
            output_path = output_dir / output_filename
            
            # Generate responses
            try:
                generate_responses_for_model(prompts_df, model_name, llm_manager, output_path)
                logger.info(f"Successfully completed {strategy_name} with {model_display_name}")
            except Exception as e:
                logger.error(f"Error processing {strategy_name} with {model_display_name}: {e}")
    
    logger.info("\n" + "="*50)
    logger.info("Response generation completed!")
    logger.info("="*50)
    
    # List generated files
    generated_files = list(output_dir.glob("*.csv"))
    logger.info(f"Generated {len(generated_files)} files:")
    for file in generated_files:
        logger.info(f"  - {file.name}")

if __name__ == "__main__":
    main() 