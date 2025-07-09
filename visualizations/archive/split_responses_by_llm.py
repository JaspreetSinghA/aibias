import pandas as pd

# Models to split
MODELS = [
    'llama-3.3-70b-versatile',
    'gpt-4',
    'claude-3-haiku-20240307',
]

# Read the main CSV
input_file = 'llm_sikh_bias_responses.csv'
df = pd.read_csv(input_file)

# Columns for output
base_columns = [
    'Prompt ID', 'Prompt Text', 'Category', 'Subcategory',
    'Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation', 'Bias Score (1-5)'
]

for model in MODELS:
    response_col = f'{model}-response'
    comments_col = f'{model}-comments'
    output_columns = base_columns.copy()
    output_columns.insert(4, 'Response')  # Insert Response after Subcategory
    output_columns.append('Comments')

    # Prepare output DataFrame
    out_df = pd.DataFrame()
    for col in base_columns:
        if col in df.columns:
            out_df[col] = df[col]
        else:
            out_df[col] = ''
    # Add Response column (model-specific, fallback to generic Response)
    if response_col in df.columns:
        responses = df[response_col].fillna("")
        if 'Response' in df.columns:
            generic_responses = df['Response'].fillna("")
            out_df['Response'] = [r if str(r).strip() else g for r, g in zip(responses, generic_responses)]
        else:
            out_df['Response'] = responses
    elif 'Response' in df.columns:
        out_df['Response'] = df['Response']
    else:
        out_df['Response'] = ''
    # Add Comments column (model-specific if exists, else blank)
    if comments_col in df.columns:
        out_df['Comments'] = df[comments_col]
    else:
        out_df['Comments'] = ''
    # Reorder columns
    out_df = out_df[['Prompt ID', 'Prompt Text', 'Category', 'Subcategory', 'Response',
                     'Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation',
                     'Bias Score (1-5)', 'Comments']]
    # Write to CSV
    out_df.to_csv(f'llm_sikh_bias_responses_{model}.csv', index=False)
    print(f'Wrote llm_sikh_bias_responses_{model}.csv') 