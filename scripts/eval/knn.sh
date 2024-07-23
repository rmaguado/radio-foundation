#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=.
torchrun --nproc_per_node=2 ./dinov2/eval/knn.py \
    --config-file ./runs/imagenet1k_vits_reg4/config.yaml \
    --pretrained-weights ./runs/imagenet1k_vits_reg4/eval/training_124999/teacher_checkpoint.pth \
    --output-dir ./runs/imagenet1k_vits_reg4/eval/training_124999/knn \
    --train-dataset ImageNet:root=./datasets/imagenet/ILSVRC2012:extra=./datasets/imagenet/extra:split=TRAIN \
    --val-dataset ImageNet:root=./datasets/imagenet/ILSVRC2012:extra=./datasets/imagenet/extra:split=VAL