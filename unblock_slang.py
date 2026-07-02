
import os
from pathlib import Path
import argparse


import os
import shutil
from pathlib import Path
import argparse

def cleanup_files(test_mode=True):
    # Added common cache folder patterns
    patterns = ["*.slangtorch", "*.lock", "*slangtorch_cache*", "__pycache__"]
    
    current_dir = Path.cwd()
    state = "TEST MODE (Dry Run)" if test_mode else "LIVE CLEANING"
    print(f"--- {state} in {current_dir} ---")
    
    deleted_count = 0
    
    for pattern in patterns:
        # rglob finds both files and directories matching the pattern
        for path in current_dir.rglob(pattern):
            try:
                path_type = "Directory" if path.is_dir() else "File"
                
                if test_mode:
                    print(f"[WOULD DELETE] {path_type}: {path}")
                else:
                    if path.is_dir():
                        shutil.rmtree(path) # Removes directory and all its contents
                    else:
                        path.unlink()       # Removes a single file
                    print(f"[DELETED] {path_type}: {path}")
                
                deleted_count += 1
            except Exception as e:
                print(f"[ERROR] Could not process {path}: {e}")

    print("---")
    print(f"Total items {'identified' if test_mode else 'removed'}: {deleted_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively delete Slangtorch files and cache dirs.")
    parser.add_argument(
        "--run", 
        action="store_true", 
        help="Actually delete the files. Without this, the script defaults to test mode."
    )
    
    args = parser.parse_args()
    
    # Defaults to True (Safety) unless --run is passed
    #cleanup_files(test_mode=not args.run)
    cleanup_files(False)

