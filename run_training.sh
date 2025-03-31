#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set PYTHONPATH to include the project root directory
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Activate virtual environment if it exists
if [ -d "${SCRIPT_DIR}/.venv" ]; then
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

# Run the training script
python "${SCRIPT_DIR}/predictor/v3/train_model_v3.py" 