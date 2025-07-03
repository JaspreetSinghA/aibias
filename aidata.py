"""
Sikh Bias Research Tool - Multi-Model Analysis
Analyzes potential biases in Large Language Models regarding Sikh-related content
"""

import pandas as pd
import time
import os
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from llm_clients import LLMManager
from config import ANALYSIS_CONFIG, FILE_PATHS, USER_RUN_CONFIG, MODEL_CONFIGS, PROMPT_CATEGORIES
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def clean_prompt(prompt_text):
    """Clean the prompt text by removing metadata (only if it appears as a whole word)"""
    if pd.isna(prompt_text):
        return ""
    # Only remove metadata if it appears as a whole word
    for meta in ["Comparative Bias", "Direct Comparison Prompts"]:
        prompt_text = re.split(rf"\\b{re.escape(meta)}\\b", prompt_text)[0].strip()
    # Remove the 'GPT' split, as it is too aggressive and not needed
    return prompt_text

def load_prompts(csv_file=FILE_PATHS['prompts_csv']):
    """Load prompts from CSV file"""
    try:
        df = pd.read_csv(csv_file, quotechar='"')
        
        # Use 'Prompt Text' column if it exists, otherwise use 'Prompt ID'
        if 'Prompt Text' in df.columns:
            df['Prompt Text'] = df['Prompt Text']
        else:
            df['Prompt Text'] = df['Prompt ID']
        
        # Filter out rows with empty prompt text
        df = df[df['Prompt Text'].notna() & (df['Prompt Text'] != '')]
        
        logger.info(f"Loaded {len(df)} prompts from {csv_file}")
        logger.debug(f"First row Prompt Text: {df.iloc[0]['Prompt Text']} (type: {type(df.iloc[0]['Prompt Text'])})")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise

def setup_response_columns(df, models):
    """Add response columns for each model if they don't exist"""
    for model in models:
        column_name = f'{model}-response'
        if column_name not in df.columns:
            df[column_name] = ""
    return df

def process_prompts_efficiently(df, llm_manager, models, delay=None):
    """Process all prompts through all models with efficient batch processing"""
    logger.info(f"Processing {len(df)} prompts through {len(models)} models")
    
    # Use configured delay or default
    if delay is None:
        delay = ANALYSIS_CONFIG['delay_between_queries']
    
    # Separate Groq models from others for batch processing
    groq_models = []
    other_models = []
    
    for model in models:
        config = llm_manager.get_model_info(model)
        if config.get('client') == 'groq':
            groq_models.append(model)
        else:
            other_models.append(model)
    
    # Process Groq models in batch for efficiency
    if groq_models:
        logger.info(f"Processing {len(groq_models)} Groq models in batch mode")
        process_groq_models_batch(df, llm_manager, groq_models)
    
    # Process other models sequentially
    if other_models:
        logger.info(f"Processing {len(other_models)} other models sequentially")
        process_other_models(df, llm_manager, other_models, delay)
    
    return df

def process_groq_models_batch(df, llm_manager, groq_models):
    """Process Groq models using batch processing for efficiency"""
    # Clean all prompts
    prompts = [clean_prompt(row['Prompt Text']) for _, row in df.iterrows()]
    prompts = [p for p in prompts if p]  # Remove empty prompts
    
    if not prompts:
        logger.warning("No valid prompts found")
        return
    
    # Use batch processing for Groq models
    try:
        batch_results = llm_manager.batch_query_models(prompts, groq_models)
        
        # Update dataframe with results
        for model_name, responses in batch_results.items():
            column_name = f'{model_name}-response'
            
            # Ensure we have the right number of responses
            if len(responses) == len(prompts):
                for i, response in enumerate(responses):
                    df.at[i, column_name] = response
            else:
                logger.warning(f"Mismatch in response count for {model_name}")
        
        logger.info(f"Batch processing completed for {len(groq_models)} Groq models")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        # Fallback to individual processing
        process_other_models(df, llm_manager, groq_models, ANALYSIS_CONFIG['delay_between_queries'])

def process_other_models(df, llm_manager, models, delay):
    """Process non-Groq models sequentially"""
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
        prompt = clean_prompt(row['Prompt Text'])
        logger.debug(f"Prompt value for row {i}: {prompt} (type: {type(prompt)})")
        
        if not prompt:
            logger.warning(f"Empty prompt at row {i}, skipping")
            continue
        
        for model in models:
            column_name = f'{model}-response'
            
            # Skip if response already exists
            if pd.notna(row[column_name]) and row[column_name] != "":
                logger.debug(f"Skipping {model} for prompt {i} - response already exists")
                continue
            
            try:
                logger.info(f"Querying {model} for prompt {i+1}/{len(df)}")
                response = llm_manager.query_model(prompt, model)
                logger.debug(f"Response type for {model} prompt {i}: {type(response)}")
                if not isinstance(response, str):
                    response = str(response)
                df.at[i, column_name] = response
                
                # Rate limiting
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error querying {model} for prompt {i}: {e}")
                df.at[i, column_name] = f"Error: {e}"
        
        # Save progress every N prompts
        if (i + 1) % ANALYSIS_CONFIG['save_progress_interval'] == 0:
            try:
                df.to_csv(FILE_PATHS['responses_csv'], index=False)
                logger.info(f"Progress saved: {i+1}/{len(df)} prompts processed")
            except Exception as e:
                logger.error(f"Error saving progress: {e}")

def estimate_total_cost(df, llm_manager, models):
    """Estimate total cost for the analysis"""
    total_cost = 0
    cost_breakdown = {}
    
    for model in models:
        # Rough estimate: assume 100 tokens per prompt
        estimated_tokens = len(df) * 100
        cost = llm_manager.estimate_cost(model, estimated_tokens)
        cost_breakdown[model] = cost
        total_cost += cost
    
    logger.info(f"Estimated total cost: ${total_cost:.4f}")
    for model, cost in cost_breakdown.items():
        logger.info(f"  {model}: ${cost:.4f}")
    
    return total_cost, cost_breakdown

def print_analysis_summary(df, llm_manager, models):
    """Print a summary of the analysis"""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total prompts processed: {len(df)}")
    print(f"Models used: {len(models)}")
    print(f"Output file: {FILE_PATHS['responses_csv']}")
    
    # Show client statistics
    client_stats = llm_manager.get_client_stats()
    if client_stats:
        print(f"\nClient Statistics:")
        for client_name, stats in client_stats.items():
            print(f"  {client_name}:")
            print(f"    Requests: {stats.get('requests_made', 0)}")
            print(f"    Errors: {stats.get('errors_encountered', 0)}")
            print(f"    Success Rate: {stats.get('success_rate', 0):.2%}")
    
    # Estimate costs
    total_cost, cost_breakdown = estimate_total_cost(df, llm_manager, models)
    
    print("="*60)

def filter_prompts_by_user_config(df):
    """Filter and sample prompts according to USER_RUN_CONFIG settings."""
    config = USER_RUN_CONFIG
    # Filter by category if specified
    if config.get('categories_to_run'):
        df = df[df['Category'].isin(config['categories_to_run'])]
    # Sample prompts per category if specified
    if config.get('prompts_per_category'):
        sampled = []
        for cat, group in df.groupby('Category'):
            n = config['prompts_per_category'].get(cat)
            if n is None:
                sampled.append(group)
            else:
                if config.get('randomize_prompts', False):
                    sampled.append(group.sample(n=min(n, len(group)), random_state=42))
                else:
                    sampled.append(group.head(n))
        df = pd.concat(sampled).reset_index(drop=True)
    return df

def main():
    """Main execution function (updated for USER_RUN_CONFIG)"""
    logger.info("Starting Sikh Bias Research Analysis with Groq Integration and user config")
    # Initialize LLM Manager with model_param_overrides if present
    llm_manager = LLMManager(model_param_overrides=USER_RUN_CONFIG.get('model_param_overrides', {}))
    # Get available models
    available_models = llm_manager.get_available_models()
    # Use only models specified in USER_RUN_CONFIG and available
    user_models = USER_RUN_CONFIG.get('models_to_run')
    if user_models:
        models_to_run = [m for m in user_models if m in available_models]
    else:
        models_to_run = list(available_models.keys())
    if not models_to_run:
        logger.error("No models selected or available. Please check your config and API keys.")
        return
    logger.info(f"Models to run: {models_to_run}")
    # Load prompts
    try:
        df = load_prompts()
    except Exception as e:
        logger.error(f"Failed to load prompts: {e}")
        return
    # Filter prompts by user config
    df = filter_prompts_by_user_config(df)
    logger.info(f"Prompts after filtering: {len(df)}")
    # Setup response columns for all selected models
    df = setup_response_columns(df, models_to_run)
    # Process prompts efficiently
    try:
        df = process_prompts_efficiently(df, llm_manager, models_to_run)
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return
    # Save final results
    try:
        df.to_csv(FILE_PATHS['responses_csv'], index=False)
        logger.info("Analysis complete! Results saved to " + FILE_PATHS['responses_csv'])
        # Print summary
        print_analysis_summary(df, llm_manager, models_to_run)
    except Exception as e:
        logger.error(f"Error saving final results: {e}")

if __name__ == "__main__":
    main()
