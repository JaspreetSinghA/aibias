"""
Test script for LLM models in Sikh Bias Research Tool
Tests connectivity and basic functionality of all configured models
"""

import os
import sys
from dotenv import load_dotenv
from llm_clients import LLMManager
from config import MODEL_CONFIGS, ANALYSIS_CONFIG

# Load environment variables
load_dotenv()

def test_single_model(llm_manager, model_name, test_prompt):
    """Test a single model with a test prompt"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    # Get model info
    model_info = llm_manager.get_model_info(model_name)
    if model_info:
        print(f"Provider: {model_info.get('provider', 'Unknown')}")
        print(f"Client: {model_info.get('client', 'Unknown')}")
        print(f"Description: {model_info.get('description', 'No description')}")
        print(f"Cost per 1K tokens: ${model_info.get('cost_per_1k_tokens', 0):.6f}")
        if model_info.get('context_window'):
            print(f"Context window: {model_info.get('context_window'):,} tokens")
    
    # Test the model
    print(f"\nTest prompt: {test_prompt}")
    print("-" * 40)
    
    try:
        response = llm_manager.query_model(test_prompt, model_name)
        print(f"Response: {response}")
        
        if response.startswith("Error:"):
            print(f"❌ FAILED: {response}")
            return False
        else:
            print("✅ SUCCESS")
            return True
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_all_models():
    """Test all available models"""
    print("🧪 LLM Model Testing Suite")
    print("=" * 60)
    
    # Initialize LLM Manager
    llm_manager = LLMManager()
    
    # Get available models
    available_models = llm_manager.get_available_models()
    
    if not available_models:
        print("❌ No models available!")
        print("\nRequired environment variables:")
        print("- OPENAI_API_KEY (for GPT models)")
        print("- ANTHROPIC_API_KEY (for Claude models)")
        print("- GOOGLE_API_KEY (for Gemini models)")
        print("- GROQ_API_KEY (for Groq models - RECOMMENDED)")
        print("\nFor local models, ensure you have sufficient GPU memory.")
        return
    
    print(f"📊 Found {len(available_models)} available models:")
    
    # Group models by client
    groq_models = []
    other_models = []
    
    for model_name, config in available_models.items():
        if config.get('client') == 'groq':
            groq_models.append(model_name)
        else:
            other_models.append(model_name)
    
    if groq_models:
        print(f"\n🚀 Groq Models ({len(groq_models)}):")
        for model in groq_models:
            print(f"  - {model}")
    
    if other_models:
        print(f"\n🔌 Other Models ({len(other_models)}):")
        for model in other_models:
            print(f"  - {model}")
    
    # Test prompt
    test_prompt = "What are the Five Ks in Sikhism and what do they represent?"
    
    print(f"\n🔍 Testing with prompt: '{test_prompt}'")
    
    # Test each model
    results = {}
    successful_models = 0
    
    for model_name in available_models.keys():
        success = test_single_model(llm_manager, model_name, test_prompt)
        results[model_name] = success
        if success:
            successful_models += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total models tested: {len(available_models)}")
    print(f"Successful: {successful_models}")
    print(f"Failed: {len(available_models) - successful_models}")
    
    if successful_models > 0:
        print(f"\n✅ Working models:")
        for model_name, success in results.items():
            if success:
                print(f"  - {model_name}")
    
    if len(available_models) - successful_models > 0:
        print(f"\n❌ Failed models:")
        for model_name, success in results.items():
            if not success:
                print(f"  - {model_name}")

def test_specific_model(model_name):
    """Test a specific model"""
    print(f"🧪 Testing specific model: {model_name}")
    
    llm_manager = LLMManager()
    
    if model_name not in MODEL_CONFIGS:
        print(f"❌ Model '{model_name}' not found in configuration")
        return
    
    available_models = llm_manager.get_available_models()
    
    if model_name not in available_models:
        print(f"❌ Model '{model_name}' not available")
        print("Check your API keys and configuration")
        return
    
    test_prompt = "What are the Five Ks in Sikhism and what do they represent?"
    test_single_model(llm_manager, model_name, test_prompt)

def show_model_info():
    """Show information about all configured models"""
    print("📚 MODEL INFORMATION")
    print("=" * 60)
    
    llm_manager = LLMManager()
    available_models = llm_manager.get_available_models()
    
    # Group by client
    models_by_client = {}
    for model_name, config in MODEL_CONFIGS.items():
        client = config.get('client', 'unknown')
        if client not in models_by_client:
            models_by_client[client] = []
        models_by_client[client].append((model_name, config))
    
    for client, models in models_by_client.items():
        print(f"\n🔌 {client.upper()} MODELS:")
        print("-" * 40)
        
        for model_name, config in models:
            status = "✅ Available" if model_name in available_models else "❌ Not Available"
            print(f"\n{model_name}: {status}")
            print(f"  Provider: {config.get('provider', 'Unknown')}")
            print(f"  Description: {config.get('description', 'No description')}")
            print(f"  Cost per 1K tokens: ${config.get('cost_per_1k_tokens', 0):.6f}")
            print(f"  Max tokens: {config.get('max_tokens', 'Unknown')}")
            print(f"  Temperature: {config.get('temperature', 'Unknown')}")
            if config.get('context_window'):
                print(f"  Context window: {config.get('context_window'):,} tokens")

def show_groq_advantages():
    """Show advantages of using Groq"""
    print("🚀 GROQ ADVANTAGES")
    print("=" * 60)
    print("✅ Single API for multiple models")
    print("✅ Ultra-fast inference (up to 1000+ tokens/second)")
    print("✅ Cost-effective pricing")
    print("✅ Built-in rate limiting and retry logic")
    print("✅ Batch processing capabilities")
    print("✅ No need to manage multiple API keys")
    print("✅ Production-ready models")
    print("✅ Large context windows (up to 131K tokens)")
    print("\n📊 Available Groq Models:")
    
    groq_models = [name for name, config in MODEL_CONFIGS.items() if config.get('client') == 'groq']
    for model in groq_models:
        config = MODEL_CONFIGS[model]
        print(f"  - {model}: {config.get('description', 'No description')}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            test_all_models()
        elif command == "test-model":
            if len(sys.argv) > 2:
                model_name = sys.argv[2]
                test_specific_model(model_name)
            else:
                print("Usage: python test_models.py test-model <model_name>")
        elif command == "info":
            show_model_info()
        elif command == "groq":
            show_groq_advantages()
        else:
            print("Unknown command. Available commands:")
            print("  test        - Test all available models")
            print("  test-model  - Test a specific model")
            print("  info        - Show model information")
            print("  groq        - Show Groq advantages")
    else:
        # Default: test all models
        test_all_models()

if __name__ == "__main__":
    main() 