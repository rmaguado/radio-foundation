#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=.
torchrun --nproc_per_node=2 ./dinov2/train/train.py \
    --config-file=./configs/lidc/vits14_reg4.yaml \
    --output-dir="./runs/lidc_vits_reg4"