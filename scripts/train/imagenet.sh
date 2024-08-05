#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=.
torchrun --nproc_per_node=4 ./dinov2/train/train.py \
    --config-file=./configs/imagenet/vits14_reg4.yaml \
    --output-dir="./runs/gradac2_imagenet1k_vits_reg4"