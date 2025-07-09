"""
Utility script to easily add new Groq models to the configuration
"""

import json
from config import MODEL_CONFIGS

def add_groq_model(model_id, provider, description, max_tokens=1000, temperature=0.3, 
                   cost_per_1k_tokens=0.0001, context_window=8192):
    """
    Add a new Groq model to the configuration
    
    Args:
        model_id: The model identifier (e.g., 'my-new-model')
        provider: The model provider (e.g., 'Meta', 'Google', 'Mistral AI')
        description: Model description
        max_tokens: Maximum tokens for completion
        temperature: Model temperature
        cost_per_1k_tokens: Cost per 1000 tokens
        context_window: Context window size in tokens
    """
    
    # Create model configuration
    model_config = {
        'client': 'groq',
        'provider': provider,
        'description': description,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'cost_per_1k_tokens': cost_per_1k_tokens,
        'context_window': context_window
    }
    
    # Add to configuration
    MODEL_CONFIGS[model_id] = model_config
    
    print(f"✅ Added model: {model_id}")
    print(f"   Provider: {provider}")
    print(f"   Description: {description}")
    print(f"   Max tokens: {max_tokens:,}")
    print(f"   Context window: {context_window:,}")
    print(f"   Cost per 1K tokens: ${cost_per_1k_tokens:.6f}")
    
    return model_config

def list_available_groq_models():
    """List all currently available Groq models"""
    groq_models = {name: config for name, config in MODEL_CONFIGS.items() 
                  if config.get('client') == 'groq'}
    
    print("🚀 Currently Available Groq Models:")
    print("=" * 50)
    
    for model_name, config in groq_models.items():
        print(f"\n{model_name}:")
        print(f"  Provider: {config.get('provider', 'Unknown')}")
        print(f"  Description: {config.get('description', 'No description')}")
        print(f"  Context window: {config.get('context_window', 'Unknown'):,} tokens")
        print(f"  Cost per 1K tokens: ${config.get('cost_per_1k_tokens', 0):.6f}")

def main():
    """Main function with example usage"""
    print("🔧 Groq Model Configuration Utility")
    print("=" * 40)
    
    # List current models
    list_available_groq_models()
    
    print("\n" + "=" * 40)
    print("Example: Adding a new model")
    print("=" * 40)
    
    # Example: Add a new model
    add_groq_model(
        model_id='example-new-model',
        provider='Example Provider',
        description='Example model for demonstration',
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_tokens=0.00015,
        context_window=16384
    )
    
    print("\n📝 To add a new model programmatically:")
    print("from add_groq_model import add_groq_model")
    print("add_groq_model('model-id', 'Provider', 'Description')")
    
    print("\n📝 To add a new model manually, edit config.py:")
    print("MODEL_CONFIGS['your-model-id'] = {")
    print("    'client': 'groq',")
    print("    'provider': 'Your Provider',")
    print("    'description': 'Your description',")
    print("    'max_tokens': 1000,")
    print("    'temperature': 0.3,")
    print("    'cost_per_1k_tokens': 0.0001,")
    print("    'context_window': 8192")
    print("}")

if __name__ == "__main__":
    main() 