"""
Modular LLM Client System for Sikh Bias Research
Supports multiple LLM providers with extensible architecture
"""

import os
import time
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import openai
import anthropic
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from groq import Groq
from config import MODEL_CONFIGS, SYSTEM_MESSAGE_TEMPLATE, ANALYSIS_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, requests_per_second: int = 5, requests_per_minute: int = 300):
        self.requests_per_second = requests_per_second
        self.requests_per_minute = requests_per_minute
        self.last_request_time = 0
        self.request_times = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        # Clean old requests (older than 1 minute)
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # Check minute limit
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                return self.wait_if_needed()
        
        # Check second limit
        time_since_last = now - self.last_request_time
        if time_since_last < 1.0 / self.requests_per_second:
            sleep_time = (1.0 / self.requests_per_second) - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_times.append(time.time())

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def query(self, prompt: str, model_name: str, **kwargs) -> str:
        """Query the model and return response"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the client is properly configured"""
        pass

class GroqClient(LLMClient):
    """Groq API client with rate limiting and batch processing"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.api_key = api_key
        self.rate_limiter = RateLimiter(
            requests_per_second=ANALYSIS_CONFIG['rate_limit_per_second'],
            requests_per_minute=ANALYSIS_CONFIG['rate_limit_per_minute']
        )
        self.request_count = 0
        self.error_count = 0
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def query(self, prompt: str, model_name: str, **kwargs) -> str:
        """Query Groq model with rate limiting and retry logic"""
        self.rate_limiter.wait_if_needed()
        
        try:
            # Get model-specific configuration
            model_config = MODEL_CONFIGS.get(model_name, {})
            
            # Prepare messages
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE_TEMPLATE},
                {"role": "user", "content": prompt}
            ]
            
            # Make API call
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=kwargs.get('temperature', model_config.get('temperature', 0.3)),
                max_tokens=kwargs.get('max_tokens', model_config.get('max_tokens', 1000)),
                top_p=kwargs.get('top_p', 0.9),
                stream=False
            )
            
            self.request_count += 1
            return response.choices[0].message.content
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Groq API error for {model_name}: {e}")
            
            # Retry logic
            max_retries = ANALYSIS_CONFIG['max_retries']
            if self.error_count <= max_retries:
                logger.info(f"Retrying {model_name} (attempt {self.error_count}/{max_retries})")
                time.sleep(2 ** self.error_count)  # Exponential backoff
                return self.query(prompt, model_name, **kwargs)
            
            return f"Error: {e}"
    
    def batch_query(self, prompts: List[str], model_name: str, **kwargs) -> List[str]:
        """Process multiple prompts in batch with rate limiting"""
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)} with {model_name}")
            result = self.query(prompt, model_name, **kwargs)
            results.append(result)
            
            # Progress update
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i+1}/{len(prompts)} prompts")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            'requests_made': self.request_count,
            'errors_encountered': self.error_count,
            'success_rate': (self.request_count - self.error_count) / max(self.request_count, 1)
        }

class OpenAIClient(LLMClient):
    """OpenAI API client"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.api_key = api_key
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def query(self, prompt: str, model_name: str, **kwargs) -> str:
        try:
            # Get model-specific configuration
            model_config = MODEL_CONFIGS.get(model_name, {})
            # Debug: log types
            logger.debug(f"Prompt type: {type(prompt)}, SYSTEM_MESSAGE_TEMPLATE type: {type(SYSTEM_MESSAGE_TEMPLATE)}")
            if not isinstance(prompt, str):
                prompt = str(prompt)
            system_message = SYSTEM_MESSAGE_TEMPLATE
            if not isinstance(system_message, str):
                system_message = str(system_message)
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get('temperature', model_config.get('temperature', 0.3)),
                max_tokens=kwargs.get('max_tokens', model_config.get('max_tokens', 1000)),
                top_p=kwargs.get('top_p', 0.9),
                frequency_penalty=kwargs.get('frequency_penalty', 0.5),
                presence_penalty=kwargs.get('presence_penalty', 0.5)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error: {e}"

class AnthropicClient(LLMClient):
    """Anthropic Claude API client"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.api_key = api_key
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def query(self, prompt: str, model_name: str, **kwargs) -> str:
        try:
            # Get model-specific configuration
            model_config = MODEL_CONFIGS.get(model_name, {})
            
            response = self.client.messages.create(
                model=model_name,
                max_tokens=kwargs.get('max_tokens', model_config.get('max_tokens', 1000)),
                temperature=kwargs.get('temperature', model_config.get('temperature', 0.3)),
                system=SYSTEM_MESSAGE_TEMPLATE,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"Error: {e}"

class GoogleClient(LLMClient):
    """Google Gemini API client"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.api_key = api_key
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def query(self, prompt: str, model_name: str, **kwargs) -> str:
        try:
            # Get model-specific configuration
            model_config = MODEL_CONFIGS.get(model_name, {})
            
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', model_config.get('temperature', 0.3)),
                    max_output_tokens=kwargs.get('max_tokens', model_config.get('max_tokens', 1000)),
                    top_p=kwargs.get('top_p', 0.9)
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Google Gemini API error: {e}")
            return f"Error: {e}"

class HuggingFaceClient(LLMClient):
    """Hugging Face local model client"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
    
    def _load_model(self, model_name: str):
        """Load a specific model and tokenizer"""
        if model_name in self.models:
            return True
            
        model_config = MODEL_CONFIGS.get(model_name, {})
        model_path = model_config.get('model_path')
        
        if not model_path:
            logger.error(f"No model path configured for {model_name}")
            return False
            
        try:
            logger.info(f"Loading model: {model_path}")
            
            # Load tokenizer
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Load model
            self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            logger.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def is_available(self) -> bool:
        # Check if at least one model is loaded
        return len(self.models) > 0
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available"""
        return model_name in self.models or self._load_model(model_name)
    
    def query(self, prompt: str, model_name: str, **kwargs) -> str:
        if not self.is_model_available(model_name):
            return f"Error: Model {model_name} not available"
        
        try:
            model_config = MODEL_CONFIGS.get(model_name, {})
            tokenizer = self.tokenizers[model_name]
            model = self.models[model_name]
            
            # Format prompt for chat models
            if "chat" in model_name.lower():
                # Use chat format for chat models
                formatted_prompt = f"<|system|>\n{SYSTEM_MESSAGE_TEMPLATE}\n<|user|>\n{prompt}\n<|assistant|>\n"
            else:
                # Use simple format for non-chat models
                formatted_prompt = f"{SYSTEM_MESSAGE_TEMPLATE}\n\nUser: {prompt}\nAssistant:"
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + model_config.get('max_tokens', 512),
                    temperature=kwargs.get('temperature', model_config.get('temperature', 0.3)),
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from response
            if formatted_prompt in response:
                response = response.split(formatted_prompt)[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Hugging Face model error for {model_name}: {e}")
            return f"Error: {e}"

class LLMManager:
    """Manager class for handling multiple LLM clients"""
    
    def __init__(self):
        self.clients: Dict[str, LLMClient] = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all available clients"""
        # OpenAI
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            self.clients['openai'] = OpenAIClient(openai_key)
            logger.info("OpenAI client initialized")
        
        # Anthropic
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            self.clients['anthropic'] = AnthropicClient(anthropic_key)
            logger.info("Anthropic client initialized")
        
        # Google
        google_key = os.getenv('GOOGLE_API_KEY')
        if google_key:
            self.clients['google'] = GoogleClient(google_key)
            logger.info("Google client initialized")
        
        # Groq
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            self.clients['groq'] = GroqClient(groq_key)
            logger.info("Groq client initialized")
        
        # Hugging Face (for local models)
        self.clients['huggingface'] = HuggingFaceClient()
        logger.info("Hugging Face client initialized")
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models based on configured clients"""
        available_models = {}
        
        for model_name, config in MODEL_CONFIGS.items():
            client_name = config['client']
            if client_name in self.clients:
                client = self.clients[client_name]
                
                # Special handling for HuggingFace models
                if client_name == 'huggingface':
                    if client.is_model_available(model_name):
                        available_models[model_name] = config
                elif client.is_available():
                    available_models[model_name] = config
        
        return available_models
    
    def query_model(self, prompt: str, model_name: str, **kwargs) -> str:
        """Query a specific model"""
        if model_name not in MODEL_CONFIGS:
            return f"Error: Model {model_name} not configured"
        
        config = MODEL_CONFIGS[model_name]
        client_name = config['client']
        
        if client_name not in self.clients:
            return f"Error: Client {client_name} not available"
        
        client = self.clients[client_name]
        
        # Special handling for HuggingFace models
        if client_name == 'huggingface':
            if not client.is_model_available(model_name):
                return f"Error: Model {model_name} not loaded"
        elif not client.is_available():
            return f"Error: Client {client_name} not properly configured"
        
        return client.query(prompt, model_name, **kwargs)
    
    def batch_query_models(self, prompts: List[str], model_names: List[str], **kwargs) -> Dict[str, List[str]]:
        """Query multiple models with multiple prompts efficiently"""
        results = {}
        
        for model_name in model_names:
            if model_name not in MODEL_CONFIGS:
                logger.warning(f"Model {model_name} not configured, skipping")
                continue
            
            config = MODEL_CONFIGS[model_name]
            client_name = config['client']
            
            if client_name not in self.clients:
                logger.warning(f"Client {client_name} not available for {model_name}, skipping")
                continue
            
            client = self.clients[client_name]
            
            # Use batch processing for Groq client
            if isinstance(client, GroqClient):
                results[model_name] = client.batch_query(prompts, model_name, **kwargs)
            else:
                # Sequential processing for other clients
                model_results = []
                for prompt in prompts:
                    result = client.query(prompt, model_name, **kwargs)
                    model_results.append(result)
                results[model_name] = model_results
        
        return results
    
    def add_model(self, model_name: str, config: Dict[str, Any]):
        """Add a new model configuration"""
        MODEL_CONFIGS[model_name] = config
        logger.info(f"Added model configuration for: {model_name}")
    
    def add_client(self, client_name: str, client: LLMClient):
        """Add a new client"""
        self.clients[client_name] = client
        logger.info(f"Added client: {client_name}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        if model_name in MODEL_CONFIGS:
            return MODEL_CONFIGS[model_name]
        return {}
    
    def estimate_cost(self, model_name: str, num_tokens: int) -> float:
        """Estimate the cost for a given number of tokens"""
        config = MODEL_CONFIGS.get(model_name, {})
        cost_per_1k = config.get('cost_per_1k_tokens', 0)
        return (num_tokens / 1000) * cost_per_1k
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get statistics from all clients"""
        stats = {}
        for client_name, client in self.clients.items():
            if hasattr(client, 'get_stats'):
                stats[client_name] = client.get_stats()
        return stats 