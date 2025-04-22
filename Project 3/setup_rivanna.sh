#!/bin/bash

# Create project directory on Rivanna
echo "Creating project directory on Rivanna..."
ssh rivanna "mkdir -p ~/Project3"

# Transfer SCRIPTS directory to Rivanna
echo "Transferring SCRIPTS to Rivanna..."
rsync -avz --progress "/Users/aymanzouari/Jarheads/Project 3/SCRIPTS/" rivanna:~/Project3/SCRIPTS/

# Create required directories on Rivanna
echo "Creating required directories on Rivanna..."
ssh rivanna "mkdir -p ~/Project3/DATA/DeepGuardDB_v1 ~/Project3/feature_extraction_results ~/Project3/feature_test_results ~/Project3/model_results"

# Make the SLURM scripts executable
echo "Making SLURM scripts executable..."
ssh rivanna "chmod +x ~/Project3/SCRIPTS/*.slurm"

# Transfer dataset to Rivanna (optional, only if needed)
echo "To transfer the dataset to Rivanna, run:"
echo "rsync -avz --progress \"/Users/aymanzouari/Jarheads/Project 3/DATA/DeepGuardDB_v1/\" rivanna:~/Project3/DATA/DeepGuardDB_v1/"

# Instructions to run the scripts on Rivanna
echo ""
echo "===== RIVANNA SETUP COMPLETE ====="
echo ""
echo "To run the scripts on Rivanna, SSH into Rivanna first:"
echo "  ssh rivanna"
echo ""
echo "Then navigate to the scripts directory:"
echo "  cd ~/Project3/SCRIPTS"
echo ""
echo "Then run the scripts in the following order:"
echo "  1. Run test feature extraction to verify everything works:"
echo "     sbatch run_test_feature_extraction.slurm"
echo ""
echo "  2. Run full feature extraction:"
echo "     sbatch run_feature_extraction.slurm"
echo ""
echo "  3. Run the model training scripts:"
echo "     sbatch run_model_training.slurm"
echo "     or run individual models:"
echo "     sbatch run_cnn_model.slurm"
echo "     sbatch run_feature_models.slurm"
echo "     sbatch run_hybrid_model.slurm"
echo ""
echo "To check the status of your jobs:"
echo "  squeue -u $USER"
echo ""
echo "To view the log files:"
echo "  cat feature_extraction_*.log"
echo "  cat model_training_*.log"
echo ""
echo "To transfer results back to your local machine, run:"
echo "  rsync -avz --progress rivanna:~/Project3/model_results/ \"/Users/aymanzouari/Jarheads/Project 3/model_results/\"" 