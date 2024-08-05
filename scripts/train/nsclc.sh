#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3
export PYTHONPATH=.
torchrun --nproc_per_node=3 ./dinov2/train/train.py \
    --config-file=./configs/nsclc/vits14_reg4.yaml \
    --output-dir="./runs/nsclc_test"