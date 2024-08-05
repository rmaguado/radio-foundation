#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3
export PYTHONPATH=.
torchrun --nproc_per_node=3 ./dinov2/train/train.py \
    --config-file=./configs/lidc/vits14_reg4.yaml \
    --output-dir="./runs/test__lidc_vits_reg4"