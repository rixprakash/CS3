#!/bin/bash
# Script to transfer model results from Rivanna to local machine

# Define local and remote paths
LOCAL_RESULTS_DIR="/Users/aymanzouari/Jarheads/Project 3/model_results"
RIVANNA_RESULTS_DIR="rivanna:~/Project3/model_results"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_RESULTS_DIR"
echo "Created local results directory: $LOCAL_RESULTS_DIR"

# Transfer CNN model results
echo "Transferring CNN model results..."
rsync -avz "$RIVANNA_RESULTS_DIR/cnn_model/" "$LOCAL_RESULTS_DIR/cnn_model/"

# Transfer hybrid model results
echo "Transferring hybrid model results..."
rsync -avz "$RIVANNA_RESULTS_DIR/hybrid_model/" "$LOCAL_RESULTS_DIR/hybrid_model/"

# Check if transfer was successful
echo "Checking transferred files..."
echo "CNN model results:"
ls -la "$LOCAL_RESULTS_DIR/cnn_model/"
echo "Hybrid model results:"
ls -la "$LOCAL_RESULTS_DIR/hybrid_model/"

echo "Transfer completed!" 