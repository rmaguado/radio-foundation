#!/bin/bash

export DHOME="$HOME/projects/radio-foundation"
export OUT="$DHOME/runs"

export JOB_NAME=$1
export RUN_NAME=$2
export CHECKPOINT_NAME=$3
export NODE=$4
export GPUS=$5
export WORKERS=$6
export LABELS=$7
export FEATURES=$8
export EPOCHS=$9

if [ "$#" -ne 9 ]; then
    echo "Usage: $0 JOB_NAME RUN_NAME CHECKPOINT_NAME NODE GPUS WORKERS LABELS FEATURES EPOCHS"
    exit 1
fi


mkdir -p $OUT/$JOB_NAME

envsubst '$JOB_NAME $RUN_NAME $CHECKPOINT_NAME $NODE $GPUS $WORKERS $OUT $LABELS $FEATURES $EPOCHS' \
    < $PWD/scripts/evaluate/ct_rate/test.template \
    > $OUT/$JOB_NAME/$JOB_NAME.run

sbatch $OUT/$JOB_NAME/$JOB_NAME.run
