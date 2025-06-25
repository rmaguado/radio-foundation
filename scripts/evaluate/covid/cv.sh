#!/bin/bash

export DHOME="$HOME/projects/radio-foundation"
export OUT="$DHOME/runs"

export JOB_NAME=$1
export RUN_NAME=$2
export CHECKPOINT_NAME=$3
export NODE=$4
export GPUS=$5
export WORKERS=$6
export FEATURES=$7
export EPOCHS=$8
export EMBED_DIM=$9

if [ "$#" -ne 9 ]; then
    echo "Usage: $0 JOB_NAME RUN_NAME CHECKPOINT_NAME NODE GPUS WORKERS FEATURES EPOCHS EMBED_DIM"
    exit 1
fi


mkdir -p $OUT/$JOB_NAME

envsubst '$JOB_NAME $RUN_NAME $CHECKPOINT_NAME $NODE $GPUS $WORKERS $OUT $FEATURES $EPOCHS $EMBED_DIM' \
    < $PWD/scripts/evaluate/covid/cv.template \
    > $OUT/$JOB_NAME/$JOB_NAME.run

sbatch $OUT/$JOB_NAME/$JOB_NAME.run
