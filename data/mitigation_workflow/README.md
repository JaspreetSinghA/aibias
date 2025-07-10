# Mitigation Workflow Organization

This directory contains the organized structure for managing LLM bias mitigation rounds.

## Directory Structure

```
data/mitigation_workflow/
├── adjusted_prompts/           # Your mitigated prompt CSVs (ready for aidata.py)
├── adjusted_responses/         # Your regraded/adjusted response CSVs
├── graded_scores/              # Graded scores from aidata.py
├── comparison_results/         # Comparison analysis between rounds
└── [round_name]_[timestamp]/   # Individual round directories
    ├── adjusted_prompts/      # Round-specific adjusted prompts
    ├── adjusted_responses/     # Round-specific adjusted responses
    ├── graded_scores/         # Round-specific graded scores
    ├── comparison_results/    # Round-specific comparisons
    └── metadata.json         # Round metadata
```

## Workflow Steps

1. **Create a new mitigation round:**
   ```bash
   python3 scripts/mitigation_workflow_manager.py create "round_name" "description"
   ```

2. **Add your adjusted prompts CSV:**
   ```bash
   python3 scripts/mitigation_workflow_manager.py add-prompts "round_id" "path/to/adjusted_prompts.csv" "strategy_name"
   ```

3. **Use aidata.py to get responses and grade them** (keep aidata.py in its current location)

4. **Add the adjusted responses CSV:**
   ```bash
   python3 scripts/mitigation_workflow_manager.py add-adjusted "round_id" "path/to/adjusted_responses.csv" "model_name"
   ```

5. **Add the graded scores CSV:**
   ```bash
   python3 scripts/mitigation_workflow_manager.py add-graded "round_id" "path/to/graded_scores.csv" "model_name"
   ```

5. **List all rounds:**
   ```bash
   python3 scripts/mitigation_workflow_manager.py list
   ```

6. **Get info about a specific round:**
   ```bash
   python3 scripts/mitigation_workflow_manager.py info "round_id"
   ```

7. **Compare two rounds:**
   ```bash
   python3 scripts/mitigation_workflow_manager.py compare "round1_id" "round2_id" "comparison_name"
   ```

## Example Workflow

```bash
# Create a new round for testing mitigation strategies
python3 scripts/mitigation_workflow_manager.py create "test_mitigation" "Testing new mitigation strategies on lowest scored prompts"

# Add your adjusted prompts (ready for aidata.py)
python3 scripts/mitigation_workflow_manager.py add-prompts "test_mitigation_20250108_143022" "data/mitigation_workflow/adjusted_prompts/instructional_strategy_prompts.csv" "instructional"

# Use aidata.py to get responses and grades

# Add the adjusted responses (output from aidata.py)
python3 scripts/mitigation_workflow_manager.py add-adjusted "test_mitigation_20250108_143022" "path/to/your_adjusted_responses.csv" "claude-3-haiku"

# Add the graded scores (output from aidata.py)
python3 scripts/mitigation_workflow_manager.py add-graded "test_mitigation_20250108_143022" "path/to/graded_scores.csv" "claude-3-haiku"

# Check what you have
python3 scripts/mitigation_workflow_manager.py info "test_mitigation_20250108_143022"
```

## Benefits

- **Organized**: Each mitigation round is kept separate with its own directory
- **Trackable**: Metadata files track when rounds were created and their status
- **Comparable**: Easy to compare different mitigation strategies
- **Versioned**: Timestamps ensure you can track the evolution of your mitigation work
- **Flexible**: Works with your existing aidata.py workflow

## Notes

- Keep `aidata.py` in its current location - this workflow manager doesn't move it
- Each round gets a unique timestamp to avoid conflicts
- You can have multiple files per round (e.g., different models or different mitigation strategies)
- The comparison functionality helps you track improvement across rounds 