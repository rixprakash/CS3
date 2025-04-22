#!/bin/bash

# This script will check the data directory structure on Rivanna

# Base directory
BASE_DIR="$HOME/Project3"
DATA_DIR="$BASE_DIR/DATA"

echo "===== CHECKING PROJECT DIRECTORY STRUCTURE ====="
echo "Base directory: $BASE_DIR"
echo "Data directory: $DATA_DIR"

# Check main directories
echo -e "\n----- Main Directories -----"
ls -la "$BASE_DIR"

# Check DATA directory
if [ -d "$DATA_DIR" ]; then
    echo -e "\n----- DATA Directory -----"
    ls -la "$DATA_DIR"
    
    # Check for DeepGuardDB_v1 directory
    DEEPGUARD_DIR="$DATA_DIR/DeepGuardDB_v1"
    if [ -d "$DEEPGUARD_DIR" ]; then
        echo -e "\n----- DeepGuardDB_v1 Directory -----"
        ls -la "$DEEPGUARD_DIR"
        
        # Check for any dataset subdirectories
        echo -e "\n----- Dataset Subdirectories -----"
        for dir in "$DEEPGUARD_DIR"/*_dataset; do
            if [ -d "$dir" ]; then
                echo "Directory: $dir"
                ls -la "$dir"
                
                # Check for real/fake subdirectories
                if [ -d "$dir/real" ]; then
                    echo "  - Real images: $(find "$dir/real" -type f | wc -l)"
                fi
                if [ -d "$dir/fake" ]; then
                    echo "  - Fake images: $(find "$dir/fake" -type f | wc -l)"
                fi
            fi
        done
    else
        echo "DeepGuardDB_v1 directory not found"
    fi
else
    echo "DATA directory not found"
fi

# Check SCRIPTS directory
SCRIPTS_DIR="$BASE_DIR/SCRIPTS"
if [ -d "$SCRIPTS_DIR" ]; then
    echo -e "\n----- SCRIPTS Directory -----"
    ls -la "$SCRIPTS_DIR"
    
    # Check main.py arguments
    echo -e "\n----- main.py Help -----"
    python "$SCRIPTS_DIR/main.py" --help
else
    echo "SCRIPTS directory not found"
fi

# Check results directories
echo -e "\n----- Results Directories -----"
for dir in "$BASE_DIR"/model_results "$BASE_DIR"/features "$BASE_DIR"/feature_extraction_results; do
    if [ -d "$dir" ]; then
        echo "Directory: $dir"
        ls -la "$dir"
    else
        echo "Directory not found: $dir"
    fi
done

echo -e "\n===== CHECK COMPLETE =====" 