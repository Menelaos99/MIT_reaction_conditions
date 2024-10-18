#!/bin/bash

# Activate the virtual environment
#source /home/students/code/MLP/venvs/tum_ai/bin/activate
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate synthesis

SCRIPT_DIR=$(dirname "$0")

# Get the current date and time
timestamp=$(date +"%Y%m%d_%H%M%S")

# Run the Python script with nohup, creating a new log file with a timestamp
nohup python "$SCRIPT_DIR/hyperparams.py" > "$SCRIPT_DIR/nohupOutputs/output_$timestamp.log" 2>&1 &

# Deactivate the virtual environment
conda deactivate
