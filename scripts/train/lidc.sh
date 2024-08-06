#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2
export PYTHONPATH=.
torchrun --nproc_per_node=2 ./dinov2/train/train.py \
    --config-file=./configs/lidc/vits14_reg4.yaml \
    --output-dir="./runs/test_lidc"