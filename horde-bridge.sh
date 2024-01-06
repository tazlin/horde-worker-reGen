#!/bin/sh
# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Build the absolute path to the Conda environment
CONDA_ENV_PATH="$SCRIPT_DIR/conda/envs/linux/lib"

# Add the Conda environment to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_ENV_PATH:$LD_LIBRARY_PATH"

./runtime.sh python -s download_models.py
./runtime.sh python -s run_worker.py $*
