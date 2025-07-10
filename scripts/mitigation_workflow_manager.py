#!/usr/bin/env python3
"""
Mitigation Workflow Manager
Helps organize and track mitigation rounds for LLM bias analysis.
"""

import os
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

class MitigationWorkflowManager:
    def __init__(self, base_dir="data/mitigation_workflow"):
        self.base_dir = Path(base_dir)
        self.adjusted_prompts_dir = self.base_dir / "adjusted_prompts"
        self.adjusted_responses_dir = self.base_dir / "adjusted_responses"
        self.graded_dir = self.base_dir / "graded_scores"
        self.comparison_dir = self.base_dir / "comparison_results"
        
        # Create directories if they don't exist
        for dir_path in [self.adjusted_prompts_dir, self.adjusted_responses_dir, self.graded_dir, self.comparison_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_mitigation_round(self, round_name, description=""):
        """Create a new mitigation round with organized subdirectories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        round_id = f"{round_name}_{timestamp}"
        
        round_dir = self.base_dir / round_id
        round_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for this round
        (round_dir / "adjusted_prompts").mkdir(exist_ok=True)
        (round_dir / "adjusted_responses").mkdir(exist_ok=True)
        (round_dir / "graded_scores").mkdir(exist_ok=True)
        (round_dir / "comparison_results").mkdir(exist_ok=True)
        
        # Create metadata file
        metadata = {
            "round_id": round_id,
            "round_name": round_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "status": "created"
        }
        
        with open(round_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Created mitigation round: {round_id}")
        print(f"Directory: {round_dir}")
        return round_id
    
    def list_mitigation_rounds(self):
        """List all mitigation rounds."""
        rounds = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                metadata_file = item / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    rounds.append(metadata)
        
        if not rounds:
            print("No mitigation rounds found.")
            return []
        
        print("Mitigation Rounds:")
        print("-" * 80)
        for round_data in sorted(rounds, key=lambda x: x["created_at"], reverse=True):
            print(f"Round ID: {round_data['round_id']}")
            print(f"Name: {round_data['round_name']}")
            print(f"Description: {round_data['description']}")
            print(f"Created: {round_data['created_at']}")
            print(f"Status: {round_data['status']}")
            print("-" * 80)
        
        return rounds
    
    def add_adjusted_prompts(self, round_id, csv_file_path, strategy_name=""):
        """Add adjusted prompts CSV to a specific round."""
        round_dir = self.base_dir / round_id
        if not round_dir.exists():
            print(f"Round {round_id} not found.")
            return False
        
        # Copy the CSV file to the round's adjusted_prompts directory
        source_path = Path(csv_file_path)
        if not source_path.exists():
            print(f"Source file {csv_file_path} not found.")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if strategy_name:
            dest_filename = f"adjusted_prompts_{strategy_name}_{timestamp}.csv"
        else:
            dest_filename = f"adjusted_prompts_{timestamp}.csv"
        
        dest_path = round_dir / "adjusted_prompts" / dest_filename
        shutil.copy2(source_path, dest_path)
        
        print(f"Added adjusted prompts: {dest_path}")
        return str(dest_path)
    
    def add_adjusted_responses(self, round_id, csv_file_path, model_name=""):
        """Add adjusted responses CSV to a specific round."""
        round_dir = self.base_dir / round_id
        if not round_dir.exists():
            print(f"Round {round_id} not found.")
            return False
        
        # Copy the CSV file to the round's adjusted_responses directory
        source_path = Path(csv_file_path)
        if not source_path.exists():
            print(f"Source file {csv_file_path} not found.")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_name:
            dest_filename = f"adjusted_responses_{model_name}_{timestamp}.csv"
        else:
            dest_filename = f"adjusted_responses_{timestamp}.csv"
        
        dest_path = round_dir / "adjusted_responses" / dest_filename
        shutil.copy2(source_path, dest_path)
        
        print(f"Added adjusted responses: {dest_path}")
        return str(dest_path)
    
    def add_graded_scores(self, round_id, csv_file_path, model_name=""):
        """Add graded scores CSV to a specific round."""
        round_dir = self.base_dir / round_id
        if not round_dir.exists():
            print(f"Round {round_id} not found.")
            return False
        
        # Copy the CSV file to the round's graded_scores directory
        source_path = Path(csv_file_path)
        if not source_path.exists():
            print(f"Source file {csv_file_path} not found.")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_name:
            dest_filename = f"graded_{model_name}_{timestamp}.csv"
        else:
            dest_filename = f"graded_scores_{timestamp}.csv"
        
        dest_path = round_dir / "graded_scores" / dest_filename
        shutil.copy2(source_path, dest_path)
        
        print(f"Added graded scores: {dest_path}")
        return str(dest_path)
    
    def compare_rounds(self, round1_id, round2_id, output_name=""):
        """Compare two mitigation rounds."""
        round1_dir = self.base_dir / round1_id
        round2_dir = self.base_dir / round2_id
        
        if not round1_dir.exists() or not round2_dir.exists():
            print("One or both rounds not found.")
            return False
        
        # Find the most recent graded scores in each round
        round1_scores = list((round1_dir / "graded_scores").glob("*.csv"))
        round2_scores = list((round2_dir / "graded_scores").glob("*.csv"))
        
        if not round1_scores or not round2_scores:
            print("No graded scores found in one or both rounds.")
            return False
        
        # Use the most recent files
        round1_file = max(round1_scores, key=lambda x: x.stat().st_mtime)
        round2_file = max(round2_scores, key=lambda x: x.stat().st_mtime)
        
        # Load and compare the data
        df1 = pd.read_csv(round1_file)
        df2 = pd.read_csv(round2_file)
        
        # Create comparison report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_name:
            output_filename = f"comparison_{output_name}_{timestamp}.csv"
        else:
            output_filename = f"comparison_{round1_id}_vs_{round2_id}_{timestamp}.csv"
        
        output_path = self.comparison_dir / output_filename
        
        # Basic comparison - you can extend this based on your needs
        comparison_data = {
            "round1_file": str(round1_file),
            "round2_file": str(round2_file),
            "round1_prompts": len(df1),
            "round2_prompts": len(df2),
            "comparison_timestamp": timestamp
        }
        
        # Save comparison metadata
        comparison_metadata_path = output_path.with_suffix('.json')
        with open(comparison_metadata_path, "w") as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"Comparison created: {comparison_metadata_path}")
        return str(comparison_metadata_path)
    
    def get_round_info(self, round_id):
        """Get detailed information about a specific round."""
        round_dir = self.base_dir / round_id
        if not round_dir.exists():
            print(f"Round {round_id} not found.")
            return None
        
        metadata_file = round_dir / "metadata.json"
        if not metadata_file.exists():
            print(f"Metadata not found for round {round_id}.")
            return None
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Count files in each subdirectory
        adjusted_prompts_files = list((round_dir / "adjusted_prompts").glob("*.csv"))
        adjusted_responses_files = list((round_dir / "adjusted_responses").glob("*.csv"))
        graded_files = list((round_dir / "graded_scores").glob("*.csv"))
        
        metadata["file_counts"] = {
            "adjusted_prompts": len(adjusted_prompts_files),
            "adjusted_responses": len(adjusted_responses_files),
            "graded_scores": len(graded_files)
        }
        
        metadata["files"] = {
            "adjusted_prompts": [f.name for f in adjusted_prompts_files],
            "adjusted_responses": [f.name for f in adjusted_responses_files],
            "graded_scores": [f.name for f in graded_files]
        }
        
        print(f"Round: {round_id}")
        print(f"Name: {metadata['round_name']}")
        print(f"Description: {metadata['description']}")
        print(f"Created: {metadata['created_at']}")
        print(f"Status: {metadata['status']}")
        print(f"Files:")
        print(f"  Adjusted prompts: {len(adjusted_prompts_files)}")
        print(f"  Adjusted responses: {len(adjusted_responses_files)}")
        print(f"  Graded scores: {len(graded_files)}")
        
        return metadata

def main():
    """Main function for command-line usage."""
    manager = MitigationWorkflowManager()
    
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python mitigation_workflow_manager.py list")
        print("  python mitigation_workflow_manager.py create <round_name> [description]")
        print("  python mitigation_workflow_manager.py info <round_id>")
        print("  python mitigation_workflow_manager.py add-prompts <round_id> <csv_file> [strategy_name]")
        print("  python mitigation_workflow_manager.py add-adjusted <round_id> <csv_file> [model_name]")
        print("  python mitigation_workflow_manager.py add-graded <round_id> <csv_file> [model_name]")
        print("  python mitigation_workflow_manager.py compare <round1_id> <round2_id> [output_name]")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        manager.list_mitigation_rounds()
    
    elif command == "create":
        if len(sys.argv) < 3:
            print("Please provide a round name.")
            return
        round_name = sys.argv[2]
        description = sys.argv[3] if len(sys.argv) > 3 else ""
        manager.create_mitigation_round(round_name, description)
    
    elif command == "info":
        if len(sys.argv) < 3:
            print("Please provide a round ID.")
            return
        round_id = sys.argv[2]
        manager.get_round_info(round_id)
    
    elif command == "add-prompts":
        if len(sys.argv) < 4:
            print("Please provide round ID and CSV file path.")
            return
        round_id = sys.argv[2]
        csv_file = sys.argv[3]
        strategy_name = sys.argv[4] if len(sys.argv) > 4 else ""
        manager.add_adjusted_prompts(round_id, csv_file, strategy_name)
    
    elif command == "add-adjusted":
        if len(sys.argv) < 4:
            print("Please provide round ID and CSV file path.")
            return
        round_id = sys.argv[2]
        csv_file = sys.argv[3]
        model_name = sys.argv[4] if len(sys.argv) > 4 else ""
        manager.add_adjusted_responses(round_id, csv_file, model_name)
    
    elif command == "add-graded":
        if len(sys.argv) < 4:
            print("Please provide round ID and CSV file path.")
            return
        round_id = sys.argv[2]
        csv_file = sys.argv[3]
        model_name = sys.argv[4] if len(sys.argv) > 4 else ""
        manager.add_graded_scores(round_id, csv_file, model_name)
    
    elif command == "compare":
        if len(sys.argv) < 4:
            print("Please provide two round IDs.")
            return
        round1_id = sys.argv[2]
        round2_id = sys.argv[3]
        output_name = sys.argv[4] if len(sys.argv) > 4 else ""
        manager.compare_rounds(round1_id, round2_id, output_name)
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main() 