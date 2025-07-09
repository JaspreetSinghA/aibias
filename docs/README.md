# Sikh Bias Research Tool - Multi-Model Analysis

A comprehensive research tool for analyzing potential biases in Large Language Models (LLMs) regarding Sikh-related content. This tool supports multiple AI providers and models, enabling systematic comparison of how different AI systems represent and discuss Sikh topics.

## 🎯 Project Overview

This research tool systematically queries multiple AI models with carefully crafted prompts about Sikh topics and collects their responses for analysis. The goal is to understand how different AI models represent and discuss Sikh-related content, identify potential biases, and contribute to the growing body of AI ethics research.

### Key Features

- **🚀 Groq Integration**: Single API access to 7+ high-performance models with ultra-fast inference
- **Multi-Model Support**: Query 15+ different LLM models from major providers
- **Modular Architecture**: Easily extensible to add new models and providers
- **Comprehensive Analysis**: Support for various prompt categories and bias analysis
- **Cost Tracking**: Built-in cost estimation for API usage
- **Progress Tracking**: Automatic saving and resume capability
- **Extensible Design**: Easy to add new models, providers, and analysis features
- **Rate Limiting**: Built-in rate limiting and retry logic for reliable API calls

## 🚀 Supported Models

### 🚀 Groq Models (Recommended)
- **gemma2-9b-it** - Google's Gemma2 9B via Groq (8K context)
- **llama-3.1-8b-instant** - Meta's Llama 3.1 8B Instant (131K context)
- **llama-3.3-70b-versatile** - Meta's Llama 3.3 70B Versatile (131K context)
- **meta-llama/llama-guard-4-12b** - Meta's Llama Guard 4 12B (131K context)
- **deepseek-r1-distill-llama-70b** - DeepSeek R1 Distill Llama 70B (131K context)
- **mistral-saba-24b** - Mistral AI Saba 24B (32K context)
- **qwen/qwen3-32b** - Alibaba Cloud Qwen3 32B (131K context)

### OpenAI Models
- **GPT-3.5 Turbo** - Fast and cost-effective
- **GPT-4** - High performance model
- **GPT-4 Turbo** - Latest OpenAI model

### Anthropic Models
- **Claude 3 Haiku** - Fast and efficient
- **Claude 3 Sonnet** - Balanced performance
- **Claude 3 Opus** - Highest performance

### Google Models
- **Gemini 1.5 Pro** - Advanced reasoning
- **Gemini 1.5 Flash** - Fast and efficient
- **Gemini Pro** - Standard model

### Local Models (via Hugging Face)
- **Llama 3 8B** - Local model (requires GPU)
- **Llama 3 70B** - Large local model (requires significant GPU)

## 🚀 Why Groq?

Groq provides several advantages for this research:

- **⚡ Ultra-Fast**: Up to 1000+ tokens/second inference
- **💰 Cost-Effective**: Significantly lower costs than direct API access
- **🔧 Single API**: One API key for multiple high-performance models
- **📊 Large Context**: Up to 131K token context windows
- **🛡️ Production Ready**: Stable, reliable models for research
- **📈 Batch Processing**: Efficient handling of multiple prompts
- **🔄 Built-in Rate Limiting**: Automatic retry logic and rate management

## 📋 Prerequisites

- Python 3.8 or higher
- API keys for desired providers (Groq recommended)
- GPU (optional, for local models)
- 8GB+ RAM (16GB+ recommended for local models)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone [your-repository-url]
   cd [repository-name]
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `env_example.txt` to `.env`
   - Fill in your API keys:
   ```bash
   cp env_example.txt .env
   # Edit .env with your API keys
   ```

## 🔑 API Key Setup

### Recommended: Groq API Key
1. **Groq API Key** (RECOMMENDED - for all Groq models)
   - Get from: https://console.groq.com/
   - Add to `.env`: `GROQ_API_KEY=your_key_here`
   - This single key gives access to 7+ high-performance models

### Alternative: Individual Provider Keys
1. **OpenAI API Key** (for GPT models)
   - Get from: https://platform.openai.com/api-keys
   - Add to `.env`: `OPENAI_API_KEY=your_key_here`

2. **Anthropic API Key** (for Claude models)
   - Get from: https://console.anthropic.com/
   - Add to `.env`: `ANTHROPIC_API_KEY=your_key_here`

3. **Google API Key** (for Gemini models)
   - Get from: https://makersuite.google.com/app/apikey
   - Add to `.env`: `GOOGLE_API_KEY=your_key_here`

4. **Hugging Face Token** (optional, for gated models)
   - Get from: https://huggingface.co/settings/tokens
   - Add to `.env`: `HUGGINGFACE_TOKEN=your_token_here`

## 📁 Project Structure

```
projects/
  scripts/                # All Python scripts/utilities
  data/                   # All CSVs and raw data files
  reports/                # All .txt reports and logs
  docs/                   # All documentation and guides
  config/                 # Configuration files (config.py, llm_clients.py)
  visualizations/         # All output plots, images, and analysis folders
  __pycache__/
  venv/, myenv/, menv/    # Virtual environments
```

- All scripts are in `scripts/`
- All data files (CSVs) are in `data/`
- All reports/logs are in `reports/`
- All documentation is in `docs/`
- All config files are in `config/`
- All output folders (improved_visualizations, fresh_visualizations, bias_analysis_plots, etc.) are in `visualizations/`
- The `archive/` folder is now inside `visualizations/`

## 🧪 Testing Your Setup

Before running the full analysis, test your configuration:

```bash
# Show Groq advantages and available models
python test_models.py groq

# Test all available models
python test_models.py test

# Test a specific model
python test_models.py test-model llama-3.3-70b-versatile

# Show model information
python test_models.py info
```

## 🚀 Usage

### Running Scripts

1. **Run any script from the `scripts/` directory.**
   - Data files are referenced as `../data/filename.csv`
   - Reports/logs as `../reports/filename.txt`
   - Config as `../config/config.py` or `../config/llm_clients.py`
   - Outputs are saved in `../visualizations/`

2. **Example:**
   ```bash
   cd scripts
   python analyze_low_scores.py
   ```

3. **If you get a FileNotFoundError, check that the file path uses the correct folder.**

### Basic Usage

1. **Prepare your prompts** in `llm_sikh_bias_prompts.csv` with columns:
   - `Prompt ID`: Unique identifier
   - `Prompt Text`: The actual prompt
   - `Category`: Prompt category (optional)
   - `Subcategory`: Prompt subcategory (optional)

2. **Run the analysis**:
   ```bash
   python aidata.py
   ```

3. **Review results** in `llm_sikh_bias_responses.csv`

4. **Analyze bias**:
   ```bash
   python analyze_results.py
   ```

### Advanced Usage

#### Custom Model Configuration

Add new models in `config.py`:

```python
MODEL_CONFIGS['my-new-model'] = {
    'client': 'groq',  # or 'openai', 'anthropic', 'google', 'huggingface'
    'provider': 'Provider Name',
    'description': 'My custom model',
    'max_tokens': 1000,
    'temperature': 0.3,
    'cost_per_1k_tokens': 0.01,
    'context_window': 8192  # for Groq models
}
```

#### Custom Analysis Parameters

Modify `ANALYSIS_CONFIG` in `config.py`:

```python
ANALYSIS_CONFIG = {
    'delay_between_queries': 0.1,  # Reduced for Groq's speed
    'save_progress_interval': 5,   # Save more frequently
    'max_retries': 5,              # More retries
    'timeout': 60,                 # Longer timeout
    'batch_size': 10,              # Larger batches
    'rate_limit_per_minute': 300,  # Groq rate limit
    'rate_limit_per_second': 5     # Conservative rate limiting
}
```

## 📊 Output Format

The tool generates `llm_sikh_bias_responses.csv` containing:

- Original prompt data
- Response columns for each model (e.g., `llama-3.3-70b-versatile-response`)
- Timestamps and metadata
- Error messages for failed queries

### Sample Output Structure

```csv
Prompt ID,Prompt Text,Category,llama-3.3-70b-versatile-response,gemma2-9b-it-response,...
A1,What are the Five Ks in Sikhism?,General Knowledge,"Detailed response...","Another response...",...
A2,What is the significance of the turban?,Cultural,"Response...","Response...",...
```

## 💰 Cost Estimation

The tool includes built-in cost estimation. Groq models are particularly cost-effective:

```python
# Example cost per 1K tokens
'llama-3.1-8b-instant': {'cost_per_1k_tokens': 0.00005}  # $0.00005
'gemma2-9b-it': {'cost_per_1k_tokens': 0.0001}          # $0.0001
'llama-3.3-70b-versatile': {'cost_per_1k_tokens': 0.0002} # $0.0002
```

Estimate total cost:
```python
from llm_clients import LLMManager
llm_manager = LLMManager()
cost = llm_manager.estimate_cost('llama-3.3-70b-versatile', 1000)  # $0.0002
```

## 🔧 Troubleshooting

### Common Issues

1. **"No models available"**
   - Check your API keys in `.env`
   - Verify API key permissions
   - Test individual models with `test_models.py`

2. **Rate limiting errors**
   - Groq handles rate limiting automatically
   - Increase `delay_between_queries` in config for other providers
   - Check API usage limits

3. **Groq API errors**
   - Verify your Groq API key is correct
   - Check Groq service status
   - Ensure you have sufficient credits

4. **Memory issues with local models**
   - Use Groq models instead (no local memory required)
   - Use smaller models (8B instead of 70B)
   - Increase system RAM

### Getting Help

- Run `python test_models.py groq` to see Groq advantages
- Run `python test_models.py info` to check model availability
- Check logs for detailed error messages
- Verify API key permissions and quotas

## 🔄 Extending the Tool

### Adding New Models

1. **Add model configuration** in `config.py`
2. **Implement client** in `llm_clients.py` (if new provider)
3. **Test** with `test_models.py`

### Adding New Analysis Features

1. **Extend prompt categories** in `config.py`
2. **Add analysis functions** in new modules
3. **Update main script** to include new features

## 📈 Research Applications

This tool is designed for:

- **Academic Research**: Systematic bias analysis across multiple models
- **AI Ethics Studies**: Comparative analysis of model behavior
- **Cultural Representation Research**: Focused analysis of specific communities
- **Model Evaluation**: Comprehensive testing of AI systems

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Your chosen license]

## 🙏 Acknowledgments

- Groq for providing fast, cost-effective access to multiple LLMs
- OpenAI, Anthropic, Google, and Meta for providing model access
- The Sikh community for inspiration and guidance
- Contributors to AI ethics and bias research
- Open source community for supporting tools and libraries

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the configuration documentation

---

**Note**: This tool is designed for research purposes. Please use responsibly and in accordance with API terms of service and ethical guidelines. Groq integration provides the most efficient and cost-effective way to access multiple high-performance models for bias research.
