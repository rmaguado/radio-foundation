#!/bin/bash

export DHOME="$HOME/projects/radio-foundation"
export OUT="$DHOME/runs"

export JOB_NAME=$1
export RUN_NAME=$2
export CHECKPOINT_NAME=$3
export NODE=$4
export WORKERS=$5
export LABELS=$6
export FEATURES=$7
export EPOCHS=$8
export EMBED_DIM=$9

export GPUS=1

if [ "$#" -ne 9 ]; then
    echo "Usage: $0 JOB_NAME RUN_NAME CHECKPOINT_NAME NODE GPUS WORKERS LABELS FEATURES EPOCHS"
    exit 1
fi


mkdir -p $OUT/$JOB_NAME

envsubst '$JOB_NAME $RUN_NAME $CHECKPOINT_NAME $NODE $GPUS $WORKERS $OUT $LABELS $FEATURES $EPOCHS $EMBED_DIM' \
    < $PWD/scripts/evaluate/ct_rate/cv.template \
    > $OUT/$JOB_NAME/$JOB_NAME.run

sbatch $OUT/$JOB_NAME/$JOB_NAME.run
