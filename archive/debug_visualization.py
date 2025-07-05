import pandas as pd
import glob
import os

def debug_csv_loading():
    """Debug the CSV loading process"""
    print("DEBUGGING CSV LOADING")
    print("=" * 50)
    
    # Test the pattern
    csv_pattern = 'llm_sikh_bias_responses_*.csv'
    csv_files = glob.glob(csv_pattern)
    
    print(f"Pattern: {csv_pattern}")
    print(f"Found files: {csv_files}")
    print(f"Number of files: {len(csv_files)}")
    
    all_data = []
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            
            # Extract model name from filename
            model_name = csv_file.replace('llm_sikh_bias_responses_', '').replace('.csv', '')
            print(f"  Extracted model name: {model_name}")
            
            # Check if Model column exists
            if 'Model' in df.columns:
                print(f"  Model column exists, unique values: {df['Model'].unique()}")
            else:
                print(f"  No Model column, adding one")
                df['Model'] = model_name
            
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            all_data.append(df)
            
        except Exception as e:
            print(f"  Error: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nCombined data:")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Models found: {combined_df['Model'].unique()}")
        print(f"  Model counts:")
        for model in combined_df['Model'].unique():
            count = len(combined_df[combined_df['Model'] == model])
            print(f"    {model}: {count} rows")
    else:
        print("No data loaded successfully")

if __name__ == "__main__":
    debug_csv_loading() 