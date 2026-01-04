"""
Script to fix sys.path hacks in Jupyter notebooks.

Replaces fragile checks like:
    sys.path.append('..')
    sys.path.append('../src')

With a robust project root finder:
    project_root = Path.cwd().resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
"""

import json
import re
from pathlib import Path

def get_robust_import_block():
    return [
        "import sys\n",
        "import os\n",
        "from pathlib import Path\n",
        "\n",
        "# Add project root to sys.path if not already present\n",
        "# (Robust against running from root or notebooks/ dir)\n",
        "current_dir = Path.cwd().resolve()\n",
        "if (current_dir / 'src').exists():\n",
        "    project_root = current_dir\n",
        "elif (current_dir.parent / 'src').exists():\n",
        "    project_root = current_dir.parent\n",
        "else:\n",
        "    # Fallback to parent\n",
        "    project_root = Path.cwd().resolve().parent\n",
        "\n",
        "if str(project_root) not in sys.path:\n",
        "    sys.path.insert(0, str(project_root))\n"
    ]

def fix_notebook(file_path):
    print(f"Processing {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ Error decoding {file_path}")
        return False

    modified = False
    
    # Check first few cells for imports
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
            
        source = cell.get('source', [])
        
        # Look for sys.path hacks
        has_sys_path_hack = any(
            'sys.path.append' in line or 'sys.path.insert' in line 
            for line in source
        )
        
        if has_sys_path_hack:
            print(f"  Found sys.path hack in cell.")
            
            # Replace the entire cell content with robust block + imports
            # But wait, we don't want to lose other imports in that cell.
            # Strategy: Replace ONLY the sys.path lines with the new block? 
            # Or just prepend the block and remove the specific hack lines?
            
            new_source = []
            block_inserted = False
            
            for line in source:
                if 'sys.path.append' in line or 'sys.path.insert' in line:
                    # Replace the first occurrence with the block
                    if not block_inserted:
                        new_source.extend(get_robust_import_block())
                        block_inserted = True
                elif 'import sys' in line:
                    continue # Already in block
                elif 'from pathlib import Path' in line:
                    continue # Already in block
                else:
                    new_source.append(line)
            
            # Update cell
            cell['source'] = new_source
            modified = True
            
        # Also fix direct imports if they assume local src
        # e.g. "from src.xxx" is fine if sys.path is correct.
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1) # Standard ipynb indent is usually 1 or 2
        print(f"✅ Fixed {file_path}")
        return True
    else:
        print(f"  No changes needed.")
        return False

def main():
    root = Path(__file__).parent.parent
    notebooks_dir = root / 'notebooks'
    
    count = 0
    for file_path in notebooks_dir.glob('*.ipynb'):
        if fix_notebook(file_path):
            count += 1
            
    print(f"\nTotal notebooks fixed: {count}")

if __name__ == "__main__":
    main()
