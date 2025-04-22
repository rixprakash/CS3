#!/bin/bash

# Script to submit all jobs with dependencies
echo "Submitting all jobs for AI image detection project"

# Make the script executable
chmod +x run_improved_feature_extraction.slurm
chmod +x run_improved_feature.slurm
chmod +x run_improved_cnn.slurm
chmod +x run_improved_hybrid.slurm
chmod +x run_improved_main.slurm

# Submit feature extraction job first
echo "Submitting feature extraction job..."
feature_job=$(sbatch run_improved_feature_extraction.slurm | awk '{print $4}')
echo "Feature extraction job submitted with ID: $feature_job"

# Submit feature model job with dependency on feature extraction
echo "Submitting feature model job (depends on feature extraction)..."
feature_model_job=$(sbatch --dependency=afterok:$feature_job run_improved_feature.slurm | awk '{print $4}')
echo "Feature model job submitted with ID: $feature_model_job"

# Submit CNN model job (can run independently)
echo "Submitting CNN model job..."
cnn_job=$(sbatch run_improved_cnn.slurm | awk '{print $4}')
echo "CNN model job submitted with ID: $cnn_job"

# Submit hybrid model job with dependency on feature extraction
echo "Submitting hybrid model job (depends on feature extraction)..."
hybrid_job=$(sbatch --dependency=afterok:$feature_job run_improved_hybrid.slurm | awk '{print $4}')
echo "Hybrid model job submitted with ID: $hybrid_job"

# Optionally, submit the all-in-one job after everything else is done
# echo "Submitting all-in-one job (will run after all others complete)..."
# all_deps="${feature_job},${feature_model_job},${cnn_job},${hybrid_job}"
# all_job=$(sbatch --dependency=afterok:$all_deps run_improved_main.slurm | awk '{print $4}')
# echo "All-in-one job submitted with ID: $all_job"

echo "All jobs submitted. Use 'squeue -u $USER' to check job status."
echo "Log files will be created in the current directory."

# Print expected job completion order
echo ""
echo "Expected job execution order:"
echo "1. Feature Extraction (Job ID: $feature_job)"
echo "2. Feature Model and CNN Model (Job IDs: $feature_model_job, $cnn_job) - running in parallel"
echo "3. Hybrid Model (Job ID: $hybrid_job) - after feature extraction"
# echo "4. All-in-one Model (Job ID: $all_job) - after all other jobs"
echo ""
echo "Use 'scancel <job_id>' to cancel a job if needed." 