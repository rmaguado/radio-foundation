#!/bin/bash

mkdir -p logs

export JOB_NAME="multi_abnormality_test"
export RUN_NAME="vitb-14.10"
export CHECKPOINT_NAME="training_319999"
export EMBED_DIM=768

envsubst < job_template.slurm > job_filled.slurm

sbatch job_filled.slurm
