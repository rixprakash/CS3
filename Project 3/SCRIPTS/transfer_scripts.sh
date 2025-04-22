#!/bin/bash

# Define paths
LOCAL_SCRIPTS_DIR="/Users/aymanzouari/Jarheads/Project 3/SCRIPTS"
RIVANNA_SCRIPTS_DIR="rivanna:~/Project3/SCRIPTS"

# Display script information
echo "===== Transferring updated SLURM scripts to Rivanna ====="
echo "Local scripts directory: $LOCAL_SCRIPTS_DIR"
echo "Rivanna scripts directory: $RIVANNA_SCRIPTS_DIR"

# List scripts to transfer
SCRIPTS=(
  "run_improved_main.slurm"
  "run_cnn_model.slurm"
  "run_hybrid_model.slurm"
  "run_feature_extraction.slurm"
  "run_feature_models.slurm"
)

# Transfer each script
for script in "${SCRIPTS[@]}"; do
  echo "Transferring $script..."
  scp "$LOCAL_SCRIPTS_DIR/$script" "$RIVANNA_SCRIPTS_DIR/"
  
  if [ $? -eq 0 ]; then
    echo "  Success: $script transferred"
  else
    echo "  Error: Failed to transfer $script"
  fi
done

# Make scripts executable on Rivanna
echo "Making scripts executable on Rivanna..."
ssh rivanna "chmod +x ~/Project3/SCRIPTS/*.slurm"

echo "===== Transfer complete ====="
echo "You can now run the updated scripts on Rivanna with the following commands:"
echo "ssh rivanna"
echo "cd ~/Project3/SCRIPTS"
echo "sbatch run_feature_extraction.slurm"
echo "sbatch run_cnn_model.slurm"
echo "sbatch run_hybrid_model.slurm"
echo "sbatch run_feature_models.slurm" 