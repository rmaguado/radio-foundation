#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2
export PYTHONPATH=.
torchrun --nproc_per_node=2 ./dinov2/train/train.py \
    --config-file=./configs/ct_collection/vits14_reg4.yaml \
    --output-dir="./runs/combined_test"