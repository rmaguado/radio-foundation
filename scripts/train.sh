#!/bin/bash

# Default values
DEFAULT_DEVICES="0,1"
DEFAULT_OUTPUT_DIR="./runs/default"
DEFAULT_CONFIG_FILE="./configs/vitb.yaml"

# Function to print help message
print_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --devices    Comma-separated list of device IDs for CUDA_VISIBLE_DEVICES (default: $DEFAULT_DEVICES)"
    echo "  --config     Path to the config file (default: $DEFAULT_CONFIG_FILE)"
    echo "  --output     Directory where the output will be stored (default: $DEFAULT_OUTPUT_DIR)"
    echo "  --help       Display this help message"
    exit 0
}

# Parsing named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --devices) DEVICES="$2"; shift ;;
        --config) CONFIG_PATH="$2"; shift ;;
        --output) OUTPUT_DIR="$2"; shift ;;
        --help) print_help ;;  # Handle --help option
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Use default values if variables are not set
DEVICES=${DEVICES:-$DEFAULT_DEVICES}
CONFIG_PATH=${CONFIG_PATH:-$DEFAULT_CONFIG_FILE}
OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}

# Export environment variables
export CUDA_VISIBLE_DEVICES=$DEVICES
export PYTHONPATH=.
export OMP_NUM_THREADS=10

# Run the torchrun command with the specified or default arguments
torchrun --nproc_per_node=$(echo $DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') ./dinov2/train/train.py \
    --config-file=$CONFIG_PATH \
    --output-dir=$OUTPUT_DIR
