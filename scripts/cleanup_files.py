import os
import glob
from datetime import datetime

def analyze_project_files():
    """Analyze project files and suggest cleanup options"""
    print("PROJECT FILE ANALYSIS")
    print("=" * 50)
    
    # Get all files in current directory
    all_files = []
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and virtual environments
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'myenv', 'menv', '__pycache__']]
        for file in files:
            if not file.startswith('.'):
                all_files.append(os.path.join(root, file))
    
    # Categorize files
    categories = {
        'Python Scripts': [],
        'CSV Data Files': [],
        'Visualization Outputs': [],
        'Documentation': [],
        'Configuration': [],
        'Environment Files': [],
        'Debug Files': [],
        'Old/Backup Files': []
    }
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        if filename.endswith('.py'):
            if 'debug' in filename.lower():
                categories['Debug Files'].append(file_path)
            else:
                categories['Python Scripts'].append(file_path)
        elif filename.endswith('.csv'):
            categories['CSV Data Files'].append(file_path)
        elif filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            if 'visualization' in file_path.lower() or 'analysis' in file_path.lower():
                categories['Visualization Outputs'].append(file_path)
            else:
                categories['Old/Backup Files'].append(file_path)
        elif filename.endswith('.md') or filename.endswith('.txt'):
            categories['Documentation'].append(file_path)
        elif filename in ['requirements.txt', 'config.py', '.env']:
            categories['Configuration'].append(file_path)
        elif filename.startswith('.env') or filename.endswith('.env'):
            categories['Environment Files'].append(file_path)
        elif 'old' in filename.lower() or 'backup' in filename.lower():
            categories['Old/Backup Files'].append(file_path)
    
    # Print analysis
    total_files = sum(len(files) for files in categories.values())
    print(f"Total files found: {total_files}")
    print()
    
    for category, files in categories.items():
        if files:
            print(f"{category} ({len(files)} files):")
            for file_path in sorted(files):
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                size_str = f"({file_size:,} bytes)" if file_size > 0 else "(size unknown)"
                print(f"  - {file_path} {size_str}")
            print()
    
    return categories

def suggest_cleanup(categories):
    """Suggest files that can be safely deleted"""
    print("CLEANUP SUGGESTIONS")
    print("=" * 50)
    
    suggestions = {
        'Safe to Delete': [],
        'Consider Deleting': [],
        'Keep': []
    }
    
    # Safe to delete
    suggestions['Safe to Delete'].extend(categories['Debug Files'])
    suggestions['Safe to Delete'].extend(categories['Old/Backup Files'])
    
    # Consider deleting (old visualization outputs)
    old_viz_files = []
    for file_path in categories['Visualization Outputs']:
        if 'fresh_visualizations' in file_path or 'bias_analysis_plots' in file_path:
            old_viz_files.append(file_path)
    suggestions['Consider Deleting'].extend(old_viz_files)
    
    # Keep
    suggestions['Keep'].extend(categories['Python Scripts'])
    suggestions['Keep'].extend(categories['CSV Data Files'])
    suggestions['Keep'].extend(categories['Configuration'])
    suggestions['Keep'].extend(categories['Documentation'])
    
    # Keep recent visualization outputs
    recent_viz = [f for f in categories['Visualization Outputs'] 
                  if f not in old_viz_files]
    suggestions['Keep'].extend(recent_viz)
    
    for category, files in suggestions.items():
        if files:
            print(f"{category} ({len(files)} files):")
            for file_path in sorted(files):
                print(f"  - {file_path}")
            print()
    
    return suggestions

def create_cleanup_script(suggestions):
    """Create a cleanup script"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = f"cleanup_script_{timestamp}.py"
    
    with open(script_name, 'w') as f:
        f.write(f"""#!/usr/bin/env python3
# Cleanup script generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Review the files below before running this script

import os
import shutil

def cleanup_files():
    \"\"\"Remove suggested files\"\"\"
    
    files_to_delete = [
""")
        
        for file_path in suggestions['Safe to Delete']:
            f.write(f'        "{file_path}",\n')
        
        f.write("""    ]
    
    files_to_consider = [
""")
        
        for file_path in suggestions['Consider Deleting']:
            f.write(f'        "{file_path}",\n')
        
        f.write("""    ]
    
    print("SAFE TO DELETE:")
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            print(f"  - {{file_path}}")
    
    print("\\nCONSIDER DELETING (review first):")
    for file_path in files_to_consider:
        if os.path.exists(file_path):
            print(f"  - {{file_path}}")
    
    print("\\nTo delete safe files, uncomment the lines below:")
    print("# for file_path in files_to_delete:")
    print("#     if os.path.exists(file_path):")
    print("#         os.remove(file_path)")
    print("#         print(f'Deleted: {{file_path}}')")
    
    f.write("""
if __name__ == "__main__":
    cleanup_files()
""")

if __name__ == "__main__":
    cleanup_files()
    
    categories = analyze_project_files()
    suggestions = suggest_cleanup(categories)
    create_cleanup_script(suggestions)
    
    print("Cleanup script created! Review it before running.")
    print("\\nRECOMMENDATIONS:")
    print("1. Delete debug files (debug_*.py)")
    print("2. Keep only recent visualization outputs")
    print("3. Archive old CSV files if needed")
    print("4. Keep all Python scripts and configuration files") 