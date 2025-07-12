# LLM Sikh Bias Mitigation Research Framework

A comprehensive framework for analyzing, mitigating, and evaluating bias in Large Language Models (LLMs) regarding Sikh-related content. This project provides tools for systematic bias detection, mitigation strategy implementation, and automated bias diagnostics using embedding-based similarity analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Strategy Organization](#strategy-organization)
- [Core Components](#core-components)
- [Workflow Management](#workflow-management)
- [Usage Examples](#usage-examples)
- [API Configuration](#api-configuration)
- [Contributing](#contributing)

## 🎯 Overview

This framework enables researchers to:
- **Detect bias** in LLM responses using semantic similarity analysis
- **Implement mitigation strategies** through prompt engineering
- **Compare effectiveness** of different approaches
- **Generate comprehensive reports** with visualizations and statistics

## 📁 Directory Structure

```
data/mitigation_workflow/
├── semantic_similarity_strategy/     # Embedding-based bias detection approach
│   ├── original_baseline_responses/  # Original (non-mitigated) LLM responses
│   ├── original_baseline_diagnostics/ # Bias analysis results for baseline responses
│   ├── baseline_analysis/            # Comprehensive analysis and visualizations
│   └── bias_comparison_*.csv         # Comparison files
├── prompt_engineering_strategy/      # Prompt modification approach
│   ├── adjusted_prompts/             # Mitigated prompts (3 strategies)
│   ├── adjusted_responses/           # LLM responses to mitigated prompts
│   └── bias_diagnostics/             # Bias analysis results for mitigated responses
├── comparison_results/               # Cross-strategy comparison results
├── graded_scores/                    # Human evaluation scores (separate approach)
└── README.md                         # This file
```

## 🔄 Strategy Organization

The framework supports **two distinct mitigation approaches**:

### **1. Semantic Similarity Strategy** (`semantic_similarity_strategy/`)
- **Purpose**: Detect bias in existing responses using embedding-based similarity analysis
- **Method**: Converts responses and bias terms to vectors, computes cosine similarity
- **Output**: Bias similarity scores, flagged responses, detailed analysis
- **Use Case**: Post-generation bias detection and analysis

### **2. Prompt Engineering Strategy** (`prompt_engineering_strategy/`)
- **Purpose**: Prevent bias through prompt modification and instruction
- **Method**: Modifies prompts with bias mitigation instructions
- **Output**: Less biased responses through better prompting
- **Use Case**: Pre-generation bias prevention

### **Strategy Comparison**
- **Semantic Similarity**: Quantitative, automated, post-generation analysis
- **Prompt Engineering**: Qualitative, manual, pre-generation prevention
- **Combined Approach**: Use both for comprehensive bias mitigation

## 🔧 Core Components

### 1. Bias Diagnostic Tool (`scripts/bias_diagnostic_tool.py`)

Automatically detects bias in LLM responses using semantic similarity analysis.

**Features:**
- Embedding-based bias detection using sentence transformers
- Configurable bias terms and similarity thresholds
- Support for batch processing of multiple files
- Detailed output with similarity scores and bias flags

**Usage:**
```bash
# Analyze single file
python3 scripts/bias_diagnostic_tool.py --input responses.csv --output analysis.csv

# Analyze directory
python3 scripts/bias_diagnostic_tool.py --input-dir responses/ --output-dir diagnostics/
```

### 2. Response Generation (`scripts/generate_mitigation_responses.py`)

Generates LLM responses for mitigated prompts using multiple models.

**Features:**
- Supports GPT-4, Claude, and Llama models
- Batch processing for multiple strategies
- Automatic file naming and organization

**Usage:**
```bash
python3 scripts/generate_mitigation_responses.py
```

### 3. Baseline Analysis (`scripts/create_baseline_analysis.py`)

Creates comprehensive analysis of baseline bias results.

**Outputs:**
- Summary tables with statistical metrics
- Bar charts showing flagged response percentages
- Histograms of bias score distributions
- Box plots for score comparisons
- Detailed statistical analysis

**Usage:**
```bash
python3 scripts/create_baseline_analysis.py
```

### 4. Comparison Analysis (`scripts/compare_baseline_mitigated.py`)

Compares baseline vs. mitigated results to assess effectiveness.

**Features:**
- Statistical comparison between strategies
- Effectiveness metrics for each approach
- Strategy ranking and recommendations

**Usage:**
```bash
python3 scripts/compare_baseline_mitigated.py
```

### 5. Enhanced Analysis Tools

**Enhanced Bias Analysis** (`scripts/enhanced_bias_analysis.py`):
- Multiple similarity thresholds (0.5, 0.6, 0.7, 0.8)
- Bias category breakdown (religious, cultural, historical, linguistic)
- Advanced statistical analysis

**Strategy Effectiveness Analyzer** (`scripts/strategy_effectiveness_analyzer.py`):
- Comparative analysis of baseline vs. mitigated responses
- Strategy ranking and effectiveness metrics
- Statistical significance testing

## 📊 Workflow Management

### Step 1: Prepare Baseline Data (Semantic Similarity Strategy)
```bash
# Clean and prepare original response files
python3 scripts/prepare_baseline_responses.py

# Run bias analysis on baseline
python3 scripts/bias_diagnostic_tool.py \
  --input-dir data/mitigation_workflow/semantic_similarity_strategy/original_baseline_responses/ \
  --output-dir data/mitigation_workflow/semantic_similarity_strategy/original_baseline_diagnostics/
```

### Step 2: Generate Mitigated Responses (Prompt Engineering Strategy)
```bash
# Generate responses for all mitigation strategies
python3 scripts/generate_mitigation_responses.py
```

### Step 3: Analyze Results
```bash
# Create comprehensive baseline analysis
python3 scripts/create_baseline_analysis.py

# Compare baseline vs. mitigated results
python3 scripts/compare_baseline_mitigated.py

# Run enhanced analysis
python3 scripts/enhanced_bias_analysis.py --input responses.csv --output-dir enhanced_analysis/

# Analyze strategy effectiveness
python3 scripts/strategy_effectiveness_analyzer.py
```

## 🧠 Semantic Similarity Strategy Details

### How It Works

The bias diagnostic tool uses semantic similarity analysis to detect bias:

1. **Embedding Generation**: Converts responses and bias terms to vector representations
2. **Similarity Calculation**: Computes cosine similarity between response and bias term embeddings
3. **Threshold Analysis**: Flags responses exceeding similarity threshold (default: 0.7)
4. **Detailed Output**: Provides similarity scores, closest bias terms, and bias flags

### Default Bias Terms

The tool includes 25 predefined bias terms covering:
- Religious stereotypes: "militant", "fanatic", "radical"
- Cultural biases: "exotic", "foreign", "other"
- Historical distortions: "violent", "aggressive", "dangerous"

### Customization

You can customize bias terms, similarity thresholds, and analysis parameters:

```python
# Custom bias terms
custom_terms = ["your_bias_term_1", "your_bias_term_2"]

# Custom threshold
threshold = 0.6  # More sensitive detection
```

## 🎨 Prompt Engineering Strategy Details

### Three Mitigation Approaches

1. **Instructional Strategy** (`instructional_strategy_prompts.csv`):
   - Direct guidance to avoid stereotypes
   - Explicit instructions for respectful responses
   - Focus on avoiding essentializing assumptions

2. **Contextual Strategy** (`contextual_strategy_prompts.csv`):
   - Broader cultural/historical context
   - Enhanced background information
   - Nuanced understanding of Sikh practices

3. **Retrieval-Based Strategy** (`retrieval_based_strategy_prompts.csv`):
   - Evidence-based responses with Sikh teachings
   - Knowledge retrieval approach
   - Authentic source material integration

## 📈 Usage Examples

### Example 1: Quick Bias Check
```bash
# Check a single response file
python3 scripts/bias_diagnostic_tool.py \
  --input my_responses.csv \
  --output bias_analysis.csv
```

### Example 2: Comprehensive Analysis
```bash
# Full workflow from baseline to comparison
python3 scripts/prepare_baseline_responses.py
python3 scripts/bias_diagnostic_tool.py --input-dir semantic_similarity_strategy/original_baseline_responses/ --output-dir semantic_similarity_strategy/original_baseline_diagnostics/
python3 scripts/create_baseline_analysis.py
python3 scripts/generate_mitigation_responses.py
python3 scripts/compare_baseline_mitigated.py
```

### Example 3: Strategy Comparison
```bash
# Compare effectiveness of different strategies
python3 scripts/compare_baseline_mitigated.py
```

### Example 4: Enhanced Analysis
```bash
# Run enhanced bias analysis with multiple thresholds
python3 scripts/enhanced_bias_analysis.py \
  --input responses.csv \
  --output-dir enhanced_analysis/ \
  --strategy semantic

# Analyze strategy effectiveness
python3 scripts/strategy_effectiveness_analyzer.py \
  --workflow-dir data/mitigation_workflow \
  --output-dir strategy_analysis
```

## ⚙️ API Configuration

### Environment Setup

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GROQ_API_KEY=your_groq_key_here
```

### Model Configuration

Edit `config/config.py` to customize model settings:

```python
MODEL_CONFIGS = {
    'gpt-4': {
        'client': 'openai',
        'model': 'gpt-4',
        'temperature': 0.7
    },
    'claude-3-haiku': {
        'client': 'anthropic', 
        'model': 'claude-3-haiku-20240307',
        'temperature': 0.7
    }
}
```

## 📊 Output Formats

### Bias Analysis CSV
```csv
prompt_id,prompt_text,category,subcategory,response,model,bias_similarity_score,closest_bias_term,bias_flag,top_3_bias_terms
COMP_01,"Explain salvation...",Comparative,Direct,"In Christianity...",gpt,0.584,militant,FALSE,"militant:0.584,fanatic:0.432,radical:0.389"
```

### Summary Table
```csv
Model,Total Responses,Flagged Responses,Flagged Percentage,Mean Bias Score,Max Bias Score
GPT,54,0,0.0,0.278,0.584
CLAUDE,54,0,0.0,0.279,0.425
LLAMA,54,0,0.0,0.273,0.458
```

## 🤝 Contributing

### Adding New Mitigation Strategies

1. Create new prompt files in `prompt_engineering_strategy/adjusted_prompts/`
2. Update the response generation script to include new strategies
3. Run analysis and comparison scripts
4. Document strategy effectiveness

### Extending Bias Detection

1. Add new bias terms to the bias diagnostic tool
2. Implement new similarity metrics
3. Create custom analysis visualizations
4. Update documentation and examples 