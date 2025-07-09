import pandas as pd
import os

# List your downloaded files and corresponding rater names and model
files = [
    ('Sikh Biases LLM - Noor - LLM#3.csv', 'Noor', 'gpt-4', ''),
    ('Gurleen - LLM#3.csv', 'Gurleen', 'gpt-4', ''),
    ('Sikh Biases LLM - Jaspreet - LLM#2.csv', 'Jaspreet', 'claude', 'claude-3-haiku-20240307'),
    ('Sikh Biases LLM - Narveer - LLM#2.csv', 'Narveer', 'claude', 'claude-3-haiku-20240307'),
    ('Sikh Biases LLM - Anu Massi - LLM#1.csv', 'Anu', 'llama', 'llama-3.3-70b-versatile'),
    ('Sikh Biases LLM - Harpreet - LLM#1.csv', 'Harpreet', 'llama', 'llama-3.3-70b-versatile')
]

for file, rater, llm_name, model_version in files:
    if not os.path.exists(file):
        print(f"File not found: {file}")
        continue
    df = pd.read_csv(file)
    df.columns = [c.strip().replace('Relevence', 'Relevance') for c in df.columns]
    # Determine response column
    if 'G4' in df.columns:
        df['Response'] = df['G4']
    elif 'CH' in df.columns:
        df['Response'] = df['CH']
    elif 'L3' in df.columns:
        df['Response'] = df['L3']
    # Fill missing columns if any
    for col in ['Prompt ID','Prompt Text','Category','Subcategory','Model','Response','Accuracy','Relevance','Fairness','Neutrality','Representation','Comments']:
        if col not in df.columns:
            df[col] = ""
    df['Model'] = llm_name if not model_version else model_version
    out_cols = ['Prompt ID','Prompt Text','Category','Subcategory','Model','Response','Accuracy','Relevance','Fairness','Neutrality','Representation','Comments']
    out_df = df[out_cols]
    # Compose output filename
    if model_version:
        out_name = f"llm_sikh_bias_responses_{rater}_{model_version}.csv"
    else:
        out_name = f"llm_sikh_bias_responses_{rater}_{llm_name}.csv"
    out_df.to_csv(out_name, index=False)
    print(f"Saved: {out_name}") 