#!/bin/bash

export DHOME="$HOME/projects/radio-foundation"
export OUT="$DHOME/runs/cache_embeddings"

export ROOT_PATH=$1
export DATASET_NAME=$2
export RUN_NAME=$3
export CHECKPOINT_NAME=$4
export DB_STORATE=$5
export NODE=$6
export GPUS=$7
export GPUTYPE=$8
export WORKERS=$9

mkdir -p $OUT/

printf "\nJob  gpus-per_node: $GPUS GPUs   Workers per GPU: $WORKERS\n\n"

envsubst '$OUT $ROOT_PATH $DATASET_NAME $RUN_NAME $CHECKPOINT_NAME $DB_STORATE $NODE $GPUS $GPUTYPE $WORKERS' \
    < $PWD/scripts/cache_embeddings/sjob.template \
    > $OUT/cache.run
sbatch $OUT/cache.run
