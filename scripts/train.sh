#!/bin/bash

# Default values
DEFAULT_DEVICES="0,1,2,3"
DEFAULT_OUTPUT_DIR="./runs/default"
DEFAULT_CONFIG_FILE="./configs/ct_collection/vits14_reg4.yaml"

# Parsing named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --devices) DEVICES="$2"; shift ;;
        --config) CONFIG_PATH="$2"; shift ;;
        --output) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Use default values if variables are not set
DEVICES=${DEVICES:-$DEFAULT_DEVICES}
CONFIG_PATH=${CONFIG_PATH:-"ct_collection/vits14_reg4.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"default"}

# Set the full config file path
CONFIG_FILE="./configs/$CONFIG_PATH"

# Set the full output directory path
FULL_OUTPUT_DIR="./runs/$OUTPUT_DIR"

# Export environment variables
export CUDA_VISIBLE_DEVICES=$DEVICES
export PYTHONPATH=.

# Run the torchrun command with the specified or default arguments
torchrun --nproc_per_node=$(echo $DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') ./dinov2/train/train.py \
    --config-file=$CONFIG_FILE \
    --output-dir=$FULL_OUTPUT_DIR
