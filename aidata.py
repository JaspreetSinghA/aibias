import openai
import pandas as pd
import time 

# Load your prompt CSV file
df = pd.read_csv('llm_sikh_bias_prompts.csv')

# Add your OpenAI API key here
openai.api_key = "your-api-key-here"

# Function to query a model
def query_model(prompt, model_name):
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

# Add empty columns if they don't exist
if 'gpt-3.5-response' not in df.columns:
    df['gpt-3.5-response'] = ""
if 'gpt-4-response' not in df.columns:
    df['gpt-4-response'] = ""

# Loop through prompts
for i, row in df.iterrows():
    if pd.isna(row['gpt-3.5-response']) or row['gpt-3.5-response'] == "":
        print(f"Querying GPT-3.5 for prompt {i+1}")
        df.at[i, 'gpt-3.5-response'] = query_model(row['Prompt'], "gpt-3.5-turbo")
        time.sleep(1)  # prevent hitting rate limits

    if pd.isna(row['gpt-4-response']) or row['gpt-4-response'] == "":
        print(f"Querying GPT-4 for prompt {i+1}")
        df.at[i, 'gpt-4-response'] = query_model(row['Prompt'], "gpt-4")
        time.sleep(1)  # prevent hitting rate limits

# Save the results
df.to_csv('llm_sikh_bias_responses.csv', index=False)
print("Responses saved to llm_sikh_bias_responses.csv")
