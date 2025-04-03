#!/bin/bash

export DHOME="$HOME/projects/radio-foundation"
export OUT="$DHOME/runs"

# NAME NODE GPUS NWORKERS CONFIG

export NAME=$1
export NODE=$2
export GPUS=$4
export WORKERS=$5
export CONFIG=$6
export ZERO=$7

if [ ! -z $5 ]; then 
  printf "\n     name | $NAME\n     node | $NODE\nresources | gpu:$GPUS workers:$WORKERS \n\n"
  mkdir -p $OUT/$NAME
  envsubst '$NAME $NODE $GPUS $WORKERS $CONFIG $OUT' < $DHOME/scripts/train/vlm/train.template > $OUT/$NAME/train.run
  sbatch $OUT/$NAME/train.run

else
  printf "\nneed NAME NODE GPUS NWORKERS CONFIG\n\n"

fi
