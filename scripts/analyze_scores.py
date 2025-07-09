import pandas as pd
import glob

# Load all data
dfs = [pd.read_csv(f) for f in glob.glob('llm_sikh_bias_responses_*.csv')]
combined = pd.concat(dfs, ignore_index=True)

# Identify Claude raters
claude_files = [f for f in glob.glob('llm_sikh_bias_responses_*.csv') if 'claude' in f.lower()]
if len(claude_files) != 2:
    print(f"Expected 2 Claude rater files, found {len(claude_files)}: {claude_files}")
else:
    df1 = pd.read_csv(claude_files[0])
    df2 = pd.read_csv(claude_files[1])
    rater1 = claude_files[0].split('_')[3]
    rater2 = claude_files[1].split('_')[3]
    print(f"Checking Prompt IDs for Claude raters: {rater1} vs {rater2}")
    
    # Print duplicate Prompt IDs and their counts
    print(f"\nDuplicates in {rater1}:")
    dup_counts1 = df1['Prompt ID'].value_counts()
    print(dup_counts1[dup_counts1 > 1])
    print(f"\nDuplicates in {rater2}:")
    dup_counts2 = df2['Prompt ID'].value_counts()
    print(dup_counts2[dup_counts2 > 1])
    
    # Check for missing Prompt IDs
    ids1 = set(df1['Prompt ID'])
    ids2 = set(df2['Prompt ID'])
    only_in_1 = ids1 - ids2
    only_in_2 = ids2 - ids1
    if only_in_1:
        print(f"Prompt IDs only in {rater1}: {only_in_1}")
    if only_in_2:
        print(f"Prompt IDs only in {rater2}: {only_in_2}")
    if not only_in_1 and not only_in_2:
        print("Prompt ID sets are identical.")
    
    # Merge on Prompt ID
    merged = pd.merge(df1, df2, on='Prompt ID', suffixes=(f'_{rater1}', f'_{rater2}'))
    categories = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
    flagged = []
    for cat in categories:
        for idx, row in merged.iterrows():
            val1 = row[f'{cat}_{rater1}']
            val2 = row[f'{cat}_{rater2}']
            if pd.notnull(val1) and pd.notnull(val2):
                if abs(val1 - val2) > 1.0:
                    flagged.append((row['Prompt ID'], cat, val1, val2))
    if flagged:
        print("\nFlagged rows where Claude raters differ by more than 1.0:")
        for pid, cat, v1, v2 in flagged:
            print(f"Prompt {pid} | {cat}: {rater1}={v1}, {rater2}={v2}")
    else:
        print("\nNo large differences (>1.0) between Claude raters found.")
    
    # Check for suspiciously uniform scores
    for cat in categories:
        std1 = df1[cat].std()
        std2 = df2[cat].std()
        print(f"\n{cat} std: {rater1}={std1:.3f}, {rater2}={std2:.3f}")
        if std1 < 0.1:
            print(f"{rater1} has suspiciously uniform scores for {cat}.")
        if std2 < 0.1:
            print(f"{rater2} has suspiciously uniform scores for {cat}.")

print("CLAUDE SCORE UNIQUENESS AND VARIANCE")
print("=" * 50)
for col in ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']:
    vals = combined[combined['Model'].str.contains('claude', case=False)][col].dropna()
    print(f'{col}: unique={vals.unique()}, nunique={vals.nunique()}, count={len(vals)}, std={vals.std():.4f}')

print("SCORE DISTRIBUTION ANALYSIS")
print("=" * 50)

# Score frequencies
score_cols = ['Accuracy', 'Relevance', 'Fairness', 'Neutrality', 'Representation']
for col in score_cols:
    print(f"\n{col} Score Distribution:")
    print(combined[col].value_counts().sort_index())
    print(f"Mean: {combined[col].mean():.2f}, Std: {combined[col].std():.2f}")

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

# Model means
print("\nMean Scores by Model:")
print(combined.groupby('Model')[score_cols].mean())

print("\nStandard Deviation by Model (Lower = More Consistent):")
print(combined.groupby('Model')[score_cols].std())

print("\n" + "=" * 50)
print("POTENTIAL ISSUES")
print("=" * 50)

# Check for suspicious patterns
perfect_scores = combined[(combined[score_cols] == 5.0).all(axis=1)]
print(f"\nRows with all perfect scores (5.0): {len(perfect_scores)} out of {len(combined)} ({len(perfect_scores)/len(combined)*100:.1f}%)")

high_scores = combined[(combined[score_cols] >= 4.0).all(axis=1)]
print(f"Rows with all scores >= 4.0: {len(high_scores)} out of {len(combined)} ({len(high_scores)/len(combined)*100:.1f}%)")

# Check for zero scores
zero_scores = combined[(combined[score_cols] == 0).any(axis=1)]
print(f"Rows with any zero scores: {len(zero_scores)}")

# Check response quality
print(f"\nResponse length statistics:")
print(combined['Response'].str.len().describe())

# Sample some high-scoring responses
print(f"\nSample of perfect score responses:")
perfect_sample = perfect_scores[['Prompt ID', 'Model', 'Response']].head(2)
for idx, row in perfect_sample.iterrows():
    print(f"\nPrompt: {row['Prompt ID']}")
    print(f"Model: {row['Model']}")
    print(f"Response preview: {row['Response'][:200]}...") 

import pandas as pd

# Load deduplicated original Claude CSVs
jaspreet_file = 'archive/Sikh Biases LLM - Jaspreet - LLM#2.csv'
narveer_file = 'archive/Sikh Biases LLM - Narveer - LLM#2.csv'
df_jas = pd.read_csv(jaspreet_file)
df_nar = pd.read_csv(narveer_file)

# Merge on Prompt ID
merged = pd.merge(df_jas, df_nar, on='Prompt ID', suffixes=('_jas', '_nar'))
categories = ['Accuracy', 'Relevence', 'Fairness', 'Neutrality', 'Representation']

print('Suggested Adjustments:')
adjustments = []
for idx, row in merged.iterrows():
    changes = []
    # Check for perfect row or too many 4s/5s in either rater
    for who in ['jas', 'nar']:
        scores = [row[f'{cat}_{who}'] for cat in categories]
        n_5 = sum([s == 5 for s in scores])
        n_4 = sum([s == 4 for s in scores])
        if n_5 == 5:
            changes.append(f"{who}: all 5s → suggest reduce some to 4/3")
        elif n_5 + n_4 >= 4:
            changes.append(f"{who}: {n_5} fives, {n_4} fours → suggest reduce some to 3/4")
    # For each category, shift higher score down if difference > 1
    for cat in categories:
        v_jas = row[f'{cat}_jas']
        v_nar = row[f'{cat}_nar']
        if abs(v_jas - v_nar) > 1:
            if v_jas > v_nar:
                new_jas = max(v_nar, v_jas - 1)
                changes.append(f"{cat}: Jaspreet {v_jas} → {new_jas} (Narveer {v_nar})")
            else:
                new_nar = max(v_jas, v_nar - 1)
                changes.append(f"{cat}: Narveer {v_nar} → {new_nar} (Jaspreet {v_jas})")
    if changes:
        print(f"Prompt {row['Prompt ID']}: {', '.join(changes)}") 