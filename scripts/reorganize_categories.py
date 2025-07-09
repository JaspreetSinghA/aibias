import pandas as pd
import os

def reorganize_categories():
    """Reorganize subcategories into structured format while preserving originals"""
    
    # Define the new structured subcategory mapping
    category_mapping = {
        'Comparative Equity Assessment': {
            'Direct Religious Comparisons': [
                'Direct Religious Comparisons'
            ],
            'Controlled Variable Testing': [
                'Controlled Variable Testing'
            ],
            'Implicit Association Testing': [
                'Implicit Association Testing'
            ]
        },
        'Knowledge-Based Bias': {
            'Historical Distortion': [
                'Historical Distortion'
            ],
            'Theological Simplification': [
                'Theological Simplification'
            ],
            'Contemporary Relevance': [
                'Contemporary Relevance'
            ]
        },
        'Linguistic Bias': {
            'Semantic Association': [
                'Semantic Association'
            ],
            'Modifier Imbalance': [
                'Modifier Imbalance'
            ],
            'Narrative Framing': [
                'Narrative Framing'
            ]
        },
        'Representational Bias': {
            'Media Representation': [
                'Media Portrayal', 'Western Media Representation'
            ],
            'Visual Identity': [
                'Visual Identity', 'Turban in Sikh Identity'
            ]
        },
        'Structural Bias': {
            'Othering Patterns': [
                'Othering Patterns'
            ],
            'Power Dynamic Misrepresentation': [
                'Power Dynamic Misrepresentation'
            ],
            'Comparative Inequity': [
                'Comparative Inequity'
            ]
        }
    }
    
    # Load the data
    csv_files = [
        '../data/llm_sikh_bias_responses.csv',
        '../data/llm_sikh_bias_responses_gpt-4.csv',
        '../data/llm_sikh_bias_responses_claude.csv',
        '../data/llm_sikh_bias_responses_llama.csv'
    ]
    
    df = None
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            print(f"Loaded data from: {csv_file}")
            break
    
    if df is None:
        print("No CSV files found!")
        return
    
    # Create new structured subcategories
    def map_to_structured_subcategory(category, original_subcategory):
        if category in category_mapping:
            for structured_subcat, original_subcats in category_mapping[category].items():
                if original_subcategory in original_subcats:
                    return structured_subcat
        return original_subcategory  # Keep original if no mapping found
    
    # Add new columns
    df['Structured_Subcategory'] = df.apply(
        lambda row: map_to_structured_subcategory(row['Category'], row['Subcategory']), 
        axis=1
    )
    df['Original_Subcategory'] = df['Subcategory']  # Preserve original
    
    # Create summary of the reorganization
    print("\nCATEGORY REORGANIZATION SUMMARY")
    print("=" * 50)
    
    for category in sorted(df['Category'].unique()):
        print(f"\n{category.upper()}:")
        print("-" * (len(category) + 1))
        
        if category in category_mapping:
            for structured_subcat, original_subcats in category_mapping[category].items():
                count = len(df[(df['Category'] == category) & 
                              (df['Structured_Subcategory'] == structured_subcat)])
                print(f"  {structured_subcat}: {count} questions")
                print(f"    (Original subcategories: {', '.join(original_subcats)})")
        else:
            # For categories not in mapping
            subcats = df[df['Category'] == category]['Subcategory'].value_counts()
            for subcat, count in subcats.items():
                print(f"  {subcat}: {count} questions")
    
    # Save reorganized data
    output_file = '../data/llm_sikh_bias_responses_reorganized.csv'
    df.to_csv(output_file, index=False)
    print(f"\nReorganized data saved to: {output_file}")
    
    # Create summary table
    summary_data = []
    for category in sorted(df['Category'].unique()):
        category_data = df[df['Category'] == category]
        structured_subcats = category_data['Structured_Subcategory'].value_counts().sort_index()
        
        for structured_subcat, count in structured_subcats.items():
            summary_data.append({
                'Category': category,
                'Structured_Subcategory': structured_subcat,
                'Question_Count': count
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = '../data/category_structure_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Category structure summary saved to: {summary_file}")
    
    return df, summary_df

def create_modification_guide():
    """Create a guide for modifying the category structure"""
    guide = """
CATEGORY STRUCTURE MODIFICATION GUIDE
====================================

To modify the category structure, edit the 'category_mapping' dictionary in reorganize_categories.py:

1. ADD NEW STRUCTURED SUBCATEGORIES:
   category_mapping['Category Name']['New Structured Subcategory'] = [
       'original_subcategory_1', 'original_subcategory_2'
   ]

2. MODIFY EXISTING MAPPINGS:
   Change the lists under each structured subcategory to include different original subcategories

3. ADD NEW CATEGORIES:
   category_mapping['New Category'] = {
       'Structured Subcategory': ['original_subcategories']
   }

4. REMOVE STRUCTURED SUBCATEGORIES:
   Delete the key-value pair from the dictionary

5. REORGANIZE QUESTIONS:
   Move original subcategories between structured subcategories as needed

EXAMPLE MODIFICATION:
category_mapping['Contemporary Issues']['New Group'] = [
    'Discrimination', 'Social Relations'
]

After making changes, run the script again to regenerate the reorganized data.
"""
    
    with open('../reports/category_modification_guide.txt', 'w') as f:
        f.write(guide)
    print("Modification guide saved to: category_modification_guide.txt")

if __name__ == "__main__":
    print("REORGANIZING CATEGORIES")
    print("=" * 30)
    
    df, summary = reorganize_categories()
    create_modification_guide()
    
    print("\nREORGANIZATION COMPLETE!")
    print("=" * 30)
    print("Files created:")
    print("1. llm_sikh_bias_responses_reorganized.csv - Data with new structure")
    print("2. category_structure_summary.csv - Summary of new structure")
    print("3. category_modification_guide.txt - Guide for modifications")
    print("\nThe original detailed subcategories are preserved in 'Original_Subcategory' column.") 