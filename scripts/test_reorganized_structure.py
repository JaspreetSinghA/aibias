#!/usr/bin/env python3
"""
Test Reorganized Structure

This script tests that the reorganized folder structure works correctly
and all paths are properly updated.
"""

import os
from pathlib import Path

def test_folder_structure():
    """Test that the reorganized folder structure exists and is correct."""
    
    base_dir = Path("data/mitigation_workflow")
    
    print("=== Testing Reorganized Folder Structure ===\n")
    
    # Test main directories exist
    required_dirs = [
        "semantic_similarity_strategy",
        "prompt_engineering_strategy", 
        "comparison_results",
        "graded_scores"
    ]
    
    print("1. Testing main directories:")
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/ - EXISTS")
        else:
            print(f"   ❌ {dir_name}/ - MISSING")
    
    # Test semantic similarity strategy structure
    print("\n2. Testing semantic_similarity_strategy structure:")
    semantic_dir = base_dir / "semantic_similarity_strategy"
    semantic_subdirs = [
        "original_baseline_responses",
        "original_baseline_diagnostics", 
        "baseline_analysis"
    ]
    
    for subdir in semantic_subdirs:
        subdir_path = semantic_dir / subdir
        if subdir_path.exists():
            print(f"   ✅ {subdir}/ - EXISTS")
            # Count files
            files = list(subdir_path.glob("*.csv")) + list(subdir_path.glob("*.png")) + list(subdir_path.glob("*.txt"))
            print(f"      📁 Contains {len(files)} files")
        else:
            print(f"   ❌ {subdir}/ - MISSING")
    
    # Test prompt engineering strategy structure
    print("\n3. Testing prompt_engineering_strategy structure:")
    prompt_dir = base_dir / "prompt_engineering_strategy"
    prompt_subdirs = [
        "adjusted_prompts",
        "adjusted_responses",
        "bias_diagnostics"
    ]
    
    for subdir in prompt_subdirs:
        subdir_path = prompt_dir / subdir
        if subdir_path.exists():
            print(f"   ✅ {subdir}/ - EXISTS")
            # Count files
            files = list(subdir_path.glob("*.csv"))
            print(f"      📁 Contains {len(files)} CSV files")
        else:
            print(f"   ❌ {subdir}/ - MISSING")
    
    # Test specific files exist
    print("\n4. Testing key files:")
    key_files = [
        "semantic_similarity_strategy/original_baseline_responses/baseline_gpt_responses.csv",
        "semantic_similarity_strategy/original_baseline_responses/baseline_claude_responses.csv", 
        "semantic_similarity_strategy/original_baseline_responses/baseline_llama_responses.csv",
        "prompt_engineering_strategy/adjusted_prompts/instructional_strategy_prompts.csv",
        "prompt_engineering_strategy/adjusted_prompts/contextual_strategy_prompts.csv",
        "prompt_engineering_strategy/adjusted_prompts/retrieval_based_strategy_prompts.csv"
    ]
    
    for file_path in key_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"   ✅ {file_path} - EXISTS")
        else:
            print(f"   ❌ {file_path} - MISSING")
    
    # Test comparison files
    print("\n5. Testing comparison files:")
    comparison_files = list(base_dir.glob("bias_comparison_*.csv"))
    if comparison_files:
        print(f"   ✅ Found {len(comparison_files)} comparison files")
        for file in comparison_files:
            print(f"      📄 {file.name}")
    else:
        print("   ⚠️  No comparison files found (this is normal if not run yet)")
    
    print("\n=== Structure Test Complete ===")

def test_script_paths():
    """Test that script paths are correctly updated."""
    
    print("\n=== Testing Script Path Updates ===\n")
    
    # Test scripts that should work with new structure
    scripts_to_test = [
        "scripts/prepare_baseline_responses.py",
        "scripts/generate_mitigation_responses.py", 
        "scripts/create_baseline_analysis.py",
        "scripts/compare_baseline_mitigated.py",
        "scripts/enhanced_bias_analysis.py",
        "scripts/strategy_effectiveness_analyzer.py"
    ]
    
    for script_path in scripts_to_test:
        if Path(script_path).exists():
            print(f"   ✅ {script_path} - EXISTS")
        else:
            print(f"   ❌ {script_path} - MISSING")
    
    print("\n=== Script Test Complete ===")

def main():
    """Main test function."""
    print("🧪 Testing Reorganized LLM Sikh Bias Mitigation Framework Structure\n")
    
    test_folder_structure()
    test_script_paths()
    
    print("\n🎯 Next Steps:")
    print("1. Run: python3 scripts/prepare_baseline_responses.py")
    print("2. Run: python3 scripts/create_baseline_analysis.py") 
    print("3. Run: python3 scripts/generate_mitigation_responses.py")
    print("4. Run: python3 scripts/compare_baseline_mitigated.py")
    print("\n📚 See README.md for detailed usage instructions")

if __name__ == "__main__":
    main() 