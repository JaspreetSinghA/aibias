import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from collections import Counter
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CSV_PATTERN = os.path.join(DATA_DIR, 'llm_sikh_bias_responses_*.csv')

def load_all_comments(csv_pattern=CSV_PATTERN):
    """
    Load and analyze comments from all CSV files
    """
    all_data = []
    
    # Find all CSV files
    csv_files = glob.glob(csv_pattern)
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Extract model name from filename
            model_name = os.path.basename(csv_file).replace('llm_sikh_bias_responses_', '').replace('.csv', '')
            
            # Add model column if it doesn't exist
            if 'Model' not in df.columns:
                df['Model'] = model_name
            
            all_data.append(df)
            print(f"Loaded comments from {csv_file}")
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_data:
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def analyze_comment_patterns(df, output_dir='comment_analysis'):
    """
    Analyze patterns in comments for qualitative insights
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Filter out empty comments
    df_with_comments = df[df['Comments'].notna() & (df['Comments'] != '')]
    
    print(f"Analyzing {len(df_with_comments)} comments across {len(df['Model'].unique())} models")
    
    # 1. Comment length analysis
    df_with_comments['Comment_Length'] = df_with_comments['Comments'].str.len()
    
    # 2. Sentiment analysis based on keywords
    positive_keywords = ['excellent', 'good', 'great', 'comprehensive', 'accurate', 'balanced', 'clear']
    negative_keywords = ['bias', 'stereotyp', 'problem', 'issue', 'concern', 'limited', 'poor']
    neutral_keywords = ['basic', 'simple', 'standard', 'typical', 'usual']
    
    def classify_sentiment(comment):
        if pd.isna(comment):
            return 'neutral'
        comment_lower = comment.lower()
        
        pos_count = sum(1 for word in positive_keywords if word in comment_lower)
        neg_count = sum(1 for word in negative_keywords if word in comment_lower)
        neu_count = sum(1 for word in neutral_keywords if word in comment_lower)
        
        if pos_count > neg_count and pos_count > neu_count:
            return 'positive'
        elif neg_count > pos_count and neg_count > neu_count:
            return 'negative'
        else:
            return 'neutral'
    
    df_with_comments['Sentiment'] = df_with_comments['Comments'].apply(classify_sentiment)
    
    # 3. Common themes analysis
    all_comments = ' '.join(df_with_comments['Comments'].dropna().astype(str))
    words = re.findall(r'\b\w+\b', all_comments.lower())
    word_freq = Counter(words)
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'}
    filtered_words = {word: count for word, count in word_freq.items() if word not in stop_words and len(word) > 2}
    
    # 4. Create visualizations
    create_comment_visualizations(df_with_comments, filtered_words, output_dir)
    
    # 5. Generate qualitative report
    generate_qualitative_report(df_with_comments, filtered_words, output_dir)
    
    return df_with_comments

def create_comment_visualizations(df, word_freq, output_dir):
    """
    Create visualizations for comment analysis
    """
    # 1. Sentiment distribution by model
    plt.figure(figsize=(12, 8))
    
    # Sentiment by model
    sentiment_by_model = df.groupby(['Model', 'Sentiment']).size().unstack(fill_value=0)
    sentiment_by_model.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Comment Sentiment Distribution by Model', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Number of Comments', fontsize=12)
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sentiment_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Comment length distribution
    plt.figure(figsize=(12, 8))
    
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]['Comment_Length']
        plt.hist(model_data, alpha=0.7, label=model, bins=20)
    
    plt.title('Comment Length Distribution by Model', fontsize=14, fontweight='bold')
    plt.xlabel('Comment Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comment_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Word frequency analysis
    plt.figure(figsize=(12, 8))
    
    # Get top 15 words
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15])
    
    plt.bar(range(len(top_words)), list(top_words.values()))
    plt.title('Most Common Words in Comments', fontsize=14, fontweight='bold')
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(range(len(top_words)), list(top_words.keys()), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/word_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Bias score vs comment sentiment
    plt.figure(figsize=(12, 8))
    
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    # Box plot of bias scores by sentiment
    sentiment_data = [df[df['Sentiment'] == sentiment]['Bias_Score'] for sentiment in ['positive', 'neutral', 'negative']]
    plt.boxplot(sentiment_data, labels=['Positive', 'Neutral', 'Negative'])
    plt.title('Bias Scores by Comment Sentiment', fontsize=14, fontweight='bold')
    plt.xlabel('Comment Sentiment', fontsize=12)
    plt.ylabel('Bias Score', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bias_score_by_sentiment.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created visualizations in {output_dir}/")

def generate_qualitative_report(df, word_freq, output_dir):
    """
    Generate a qualitative analysis report
    """
    report_file = f'{output_dir}/qualitative_analysis_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("QUALITATIVE ANALYSIS OF BIAS EVALUATION COMMENTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total comments analyzed: {len(df)}\n")
        f.write(f"Models evaluated: {len(df['Model'].unique())}\n")
        f.write(f"Comments per model: {len(df) // len(df['Model'].unique())}\n\n")
        
        f.write("SENTIMENT ANALYSIS\n")
        f.write("-" * 20 + "\n")
        sentiment_counts = df['Sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"{sentiment.capitalize()}: {count} comments ({percentage:.1f}%)\n")
        
        f.write("\nSENTIMENT BY MODEL\n")
        f.write("-" * 20 + "\n")
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            f.write(f"\n{model.upper()}:\n")
            model_sentiment = model_data['Sentiment'].value_counts()
            for sentiment, count in model_sentiment.items():
                percentage = (count / len(model_data)) * 100
                f.write(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)\n")
        
        f.write("\nCOMMENT LENGTH ANALYSIS\n")
        f.write("-" * 25 + "\n")
        f.write(f"Average comment length: {df['Comment_Length'].mean():.1f} characters\n")
        f.write(f"Median comment length: {df['Comment_Length'].median():.1f} characters\n")
        f.write(f"Shortest comment: {df['Comment_Length'].min()} characters\n")
        f.write(f"Longest comment: {df['Comment_Length'].max()} characters\n")
        
        f.write("\nMOST COMMON THEMES\n")
        f.write("-" * 20 + "\n")
        top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])
        for word, count in top_words.items():
            f.write(f"{word}: {count} occurrences\n")
        
        f.write("\nQUALITATIVE INSIGHTS\n")
        f.write("-" * 20 + "\n")
        
        # Analyze high and low bias examples
        high_bias = df[df['Bias_Score'] <= 2.5]
        low_bias = df[df['Bias_Score'] >= 4.5]
        
        f.write(f"\nHigh Bias Examples (Score ≤ 2.5): {len(high_bias)} comments\n")
        for _, row in high_bias.head(5).iterrows():
            f.write(f"  Prompt: {row['Prompt ID']} - {row['Comments']}\n")
        
        f.write(f"\nLow Bias Examples (Score ≥ 4.5): {len(low_bias)} comments\n")
        for _, row in low_bias.head(5).iterrows():
            f.write(f"  Prompt: {row['Prompt ID']} - {row['Comments']}\n")
        
        f.write("\nRECOMMENDATIONS\n")
        f.write("-" * 15 + "\n")
        f.write("1. Focus on prompts with negative sentiment comments\n")
        f.write("2. Analyze common themes in high-bias responses\n")
        f.write("3. Use positive examples as templates for improvement\n")
        f.write("4. Consider comment length as indicator of evaluator engagement\n")
        f.write("5. Develop targeted improvements based on qualitative feedback\n")
    
    print(f"Generated qualitative analysis report: {report_file}")

def create_exemplar_response_table(df, output_dir='comment_analysis'):
    """
    Create a table of exemplar responses with comments
    """
    # Calculate bias scores
    df['Bias_Score'] = (df['Fairness'] + df['Neutrality'] + df['Representation']) / 3
    
    # Find exemplar responses
    exemplars = []
    
    # High bias examples
    high_bias = df[df['Bias_Score'] <= 2.5].head(3)
    for _, row in high_bias.iterrows():
        exemplars.append({
            'Type': 'High Bias',
            'Model': row['Model'],
            'Prompt ID': row['Prompt ID'],
            'Prompt': row['Prompt Text'][:100] + '...',
            'Response': row['Response'][:150] + '...',
            'Bias Score': f"{row['Bias_Score']:.2f}",
            'Comment': row['Comments']
        })
    
    # Low bias examples
    low_bias = df[df['Bias_Score'] >= 4.5].head(3)
    for _, row in low_bias.iterrows():
        exemplars.append({
            'Type': 'Low Bias',
            'Model': row['Model'],
            'Prompt ID': row['Prompt ID'],
            'Prompt': row['Prompt Text'][:100] + '...',
            'Response': row['Response'][:150] + '...',
            'Bias Score': f"{row['Bias_Score']:.2f}",
            'Comment': row['Comments']
        })
    
    # Create exemplar table
    exemplar_df = pd.DataFrame(exemplars)
    exemplar_file = f'{output_dir}/exemplar_responses.csv'
    exemplar_df.to_csv(exemplar_file, index=False)
    
    print(f"Created exemplar response table: {exemplar_file}")
    return exemplar_df

def main():
    """
    Main function to run comment analysis
    """
    print("COMMENT ANALYSIS TOOL")
    print("=" * 30)
    
    # Load data
    df = load_all_comments()
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Analyze comments
    df_with_comments = analyze_comment_patterns(df)
    
    # Create exemplar table
    exemplar_df = create_exemplar_response_table(df_with_comments)
    
    print(f"\nComment analysis complete! Check the 'comment_analysis/' directory for results.")
    print("\nGenerated files:")
    print("- sentiment_by_model.png: Sentiment distribution")
    print("- comment_length_distribution.png: Comment length analysis")
    print("- word_frequency.png: Most common words")
    print("- bias_score_by_sentiment.png: Bias vs sentiment correlation")
    print("- qualitative_analysis_report.txt: Detailed qualitative report")
    print("- exemplar_responses.csv: High/low bias examples")

if __name__ == "__main__":
    main() 