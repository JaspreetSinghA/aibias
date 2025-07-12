#!/usr/bin/env python3
"""
Threshold Analysis for Research Paper

Analyzes different bias detection thresholds and their implications
for the LLM Sikh Bias Mitigation research paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_thresholds():
    """Analyze different threshold levels and their research implications."""
    
    # Load data
    df = pd.read_csv('data/mitigation_workflow/semantic_similarity_strategy/original_baseline_diagnostics/bias_analysis_baseline_gpt_responses.csv')
    
    print("=== THRESHOLD ANALYSIS FOR RESEARCH PAPER ===\n")
    
    # Threshold sensitivity analysis
    print("THRESHOLD SENSITIVITY ANALYSIS:")
    print("-" * 40)
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for threshold in thresholds:
        flagged = (df['bias_similarity_score'] > threshold).sum()
        percentage = (flagged / len(df)) * 100
        print(f"Threshold {threshold}: {flagged} flagged ({percentage:.1f}%)")
    
    # Category analysis
    print("\nCATEGORY ANALYSIS:")
    print("-" * 40)
    category_stats = df.groupby('category')['bias_similarity_score'].agg(['mean', 'max', 'count']).round(3)
    print(category_stats)
    
    # High-scoring responses analysis
    print("\nHIGH-SCORING RESPONSES ANALYSIS:")
    print("-" * 40)
    high_scores = df[df['bias_similarity_score'] > 0.4].sort_values('bias_similarity_score', ascending=False)
    print(f"Responses with scores > 0.4: {len(high_scores)}")
    
    for idx, row in high_scores.head(5).iterrows():
        print(f"\nScore: {row['bias_similarity_score']:.3f}")
        print(f"Category: {row['category']}")
        print(f"Closest Bias Term: {row['closest_bias_term']}")
        print(f"Prompt ID: {row['prompt_id']}")
        print(f"Response Preview: {row['response'][:100]}...")
    
    # Research implications
    print("\nRESEARCH IMPLICATIONS:")
    print("-" * 40)
    
    # Current threshold analysis
    current_flagged = (df['bias_similarity_score'] > 0.7).sum()
    print(f"Current threshold (0.7): {current_flagged} flagged responses")
    
    if current_flagged == 0:
        print("⚠️  RESEARCH CHALLENGE: No responses flagged at 0.7 threshold")
        print("   - May suggest models are genuinely unbiased")
        print("   - May suggest threshold is too conservative")
        print("   - May suggest bias terms are too extreme")
    
    # Alternative thresholds
    print("\nALTERNATIVE THRESHOLD RECOMMENDATIONS:")
    print("-" * 40)
    
    for threshold in [0.5, 0.6]:
        flagged = (df['bias_similarity_score'] > threshold).sum()
        percentage = (flagged / len(df)) * 100
        print(f"Threshold {threshold}: {flagged} responses ({percentage:.1f}%)")
        
        if flagged > 0:
            print(f"   ✅ Provides sufficient data for analysis")
        else:
            print(f"   ⚠️  Still no flagged responses")
    
    # Multi-tier approach
    print("\nMULTI-TIER DETECTION APPROACH:")
    print("-" * 40)
    print("High Bias (0.7+): Clearly problematic responses")
    print("Medium Bias (0.5-0.7): Responses needing review")
    print("Low Bias (0.3-0.5): Subtle bias patterns")
    print("Very Low Bias (0.0-0.3): Minimal bias detected")
    
    # Research recommendations
    print("\nRESEARCH PAPER RECOMMENDATIONS:")
    print("-" * 40)
    print("1. Acknowledge the conservative nature of 0.7 threshold")
    print("2. Use multiple thresholds for comprehensive analysis")
    print("3. Focus on relative differences between models/strategies")
    print("4. Include qualitative analysis of high-scoring responses")
    print("5. Discuss implications of low bias detection rates")
    print("6. Consider context-specific threshold adjustments")
    
    return df, high_scores

def create_threshold_visualization(df):
    """Create visualization showing threshold sensitivity."""
    
    thresholds = np.arange(0.2, 0.8, 0.05)
    flagged_counts = []
    
    for threshold in thresholds:
        flagged = (df['bias_similarity_score'] > threshold).sum()
        flagged_counts.append(flagged)
    
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 2, 1)
    plt.plot(thresholds, flagged_counts, 'b-o', linewidth=2, markersize=6)
    plt.axvline(x=0.7, color='red', linestyle='--', label='Current Threshold (0.7)')
    plt.axvline(x=0.5, color='orange', linestyle='--', label='Suggested Threshold (0.5)')
    plt.xlabel('Bias Threshold')
    plt.ylabel('Number of Flagged Responses')
    plt.title('Threshold Sensitivity Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Score distribution
    plt.subplot(2, 2, 2)
    plt.hist(df['bias_similarity_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0.7, color='red', linestyle='--', label='Current Threshold')
    plt.axvline(x=0.5, color='orange', linestyle='--', label='Suggested Threshold')
    plt.xlabel('Bias Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Bias Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Category comparison
    plt.subplot(2, 2, 3)
    category_means = df.groupby('category')['bias_similarity_score'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(category_means)), category_means.values, color='lightcoral', alpha=0.7)
    plt.xticks(range(len(category_means)), [cat.replace(' ', '\n') for cat in category_means.index], rotation=45)
    plt.ylabel('Mean Bias Score')
    plt.title('Mean Bias Scores by Category')
    plt.grid(True, alpha=0.3)
    
    # Threshold comparison table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    threshold_data = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        flagged = (df['bias_similarity_score'] > threshold).sum()
        percentage = (flagged / len(df)) * 100
        threshold_data.append([f'{threshold}', f'{flagged}', f'{percentage:.1f}%'])
    
    table = plt.table(cellText=threshold_data,
                     colLabels=['Threshold', 'Flagged', 'Percentage'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('Threshold Comparison Table')
    
    plt.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as: threshold_analysis.png")

if __name__ == "__main__":
    df, high_scores = analyze_thresholds()
    create_threshold_visualization(df) 