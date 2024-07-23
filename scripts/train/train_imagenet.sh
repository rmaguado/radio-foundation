#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=.
python -m torch.distributed.launch --nproc_per_node=2 ./dinov2/train/train.py \
    --config-file=./configs/vits14_reg4.yaml \
    --output-dir="./runs/imagenet1k_vits_reg4"