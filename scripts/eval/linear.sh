#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=.
torchrun --nproc_per_node=4 ./dinov2/eval/linear.py \
    --config-file ./runs/test5_imagenet1k_vits_reg4/config.yaml \
    --pretrained-weights ./runs/test5_imagenet1k_vits_reg4/eval/training_312499/teacher_checkpoint.pth \
    --output-dir ./runs/test5_imagenet1k_vits_reg4/eval/training_312499/linear \
    --train-dataset ImageNet:root=./datasets/imagenet/ILSVRC2012:extra=./datasets/imagenet/extra:split=TRAIN \
    --val-dataset ImageNet:root=./datasets/imagenet/ILSVRC2012:extra=./datasets/imagenet/extra:split=VAL