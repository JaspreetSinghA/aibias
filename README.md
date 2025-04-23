# LLM Sikh Bias Analysis Tool

This project analyzes potential biases in Large Language Models (LLMs) regarding Sikh-related content by comparing responses from different AI models.

## Project Overview

The tool systematically queries both GPT-3.5 and GPT-4 models with carefully crafted prompts about Sikh topics and collects their responses for analysis. This helps in understanding how different AI models represent and discuss Sikh-related content.

## Features

- Automated prompt processing for multiple AI models
- Support for both GPT-3.5 and GPT-4 responses
- Secure API key management using environment variables
- CSV-based prompt management
- Automated response collection and storage

## Setup Instructions

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
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```

## Project Structure

- `aidata.py`: Main script for processing prompts and collecting responses
- `llm_sikh_bias_prompts.csv`: Input file containing prompts to analyze
- `llm_sikh_bias_responses.csv`: Output file containing model responses
- `.env`: Configuration file for API keys (not tracked in git)
- `requirements.txt`: Project dependencies

## Usage

1. Prepare your prompts in `llm_sikh_bias_prompts.csv` with the following columns:
   - Prompt ID
   - Prompt Text
   - Category
   - Subcategory
   - Model
   - Response
   - Bias Score (1-5)
   - Comments

2. Run the script:
   ```bash
   python aidata.py
   ```

3. The script will:
   - Process each prompt
   - Query both GPT-3.5 and GPT-4
   - Save responses to `llm_sikh_bias_responses.csv`

## Output

The script generates `llm_sikh_bias_responses.csv` containing:
- Original prompts
- GPT-3.5 responses
- GPT-4 responses
- Additional metadata

## Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure
- The `.gitignore` file is configured to exclude sensitive information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license]

## Acknowledgments

- OpenAI for providing the API access
- Contributors and researchers in the field of AI bias analysis
