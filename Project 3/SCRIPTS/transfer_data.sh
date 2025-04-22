#!/bin/bash

# Source and destination paths
LOCAL_DATA_DIR="/Users/aymanzouari/Jarheads/Project 3/DATA/DeepGuardDB_v1"
RIVANNA_DATA_DIR="rivanna:~/Project3/DATA/DeepGuardDB_v1"

echo "Starting data transfer to Rivanna..."
echo "Source: $LOCAL_DATA_DIR"
echo "Destination: $RIVANNA_DATA_DIR"

# Create the destination directory on Rivanna
ssh rivanna "mkdir -p ~/Project3/DATA/DeepGuardDB_v1"

# Transfer the data using rsync with progress and compression
rsync -avz --progress "$LOCAL_DATA_DIR/" "$RIVANNA_DATA_DIR/"

echo "Data transfer completed. Verifying..."

# Verify the transfer by checking file counts
LOCAL_COUNT=$(find "$LOCAL_DATA_DIR" -type f | wc -l)
RIVANNA_COUNT=$(ssh rivanna "find ~/Project3/DATA/DeepGuardDB_v1 -type f | wc -l")

echo "Local file count: $LOCAL_COUNT"
echo "Rivanna file count: $RIVANNA_COUNT"

if [ "$LOCAL_COUNT" -eq "$RIVANNA_COUNT" ]; then
    echo "Transfer verification successful!"
else
    echo "Warning: File counts do not match. Please check the transfer."
fi 