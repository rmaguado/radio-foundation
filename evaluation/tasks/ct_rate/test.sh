#!/bin/bash

mkdir -p logs

export JOB_NAME="multi_abnormality_cls"
export RUN_NAME="vitb_CT-RATE"
export CHECKPOINT_NAME="training_549999"

envsubst '$JOB_NAME $RUN_NAME $CHECKPOINT_NAME' < $PWD/test.template > $PWD/$JOB_NAME.run

sbatch $JOB_NAME.run
