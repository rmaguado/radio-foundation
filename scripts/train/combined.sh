#!/bin/bash

# Default values
DEFAULT_DEVICES="0,1,2,3"
DEFAULT_CONFIG_FILE="./configs/ct_collection/vits14_reg4.yaml"
DEFAULT_OUTPUT_DIR="./runs/combined"

# Parsing named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --devices) DEVICES="$2"; shift ;;
        --config-file) CONFIG_FILE="$2"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Use default values if variables are not set
DEVICES=${DEVICES:-$DEFAULT_DEVICES}
CONFIG_FILE=${CONFIG_FILE:-$DEFAULT_CONFIG_FILE}
OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}

# Export environment variables
export CUDA_VISIBLE_DEVICES=$DEVICES
export PYTHONPATH=.

# Run the torchrun command with the specified or default arguments
torchrun --nproc_per_node=$(echo $DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') ./dinov2/train/train.py \
    --config-file=$CONFIG_FILE \
    --output-dir=$OUTPUT_DIR