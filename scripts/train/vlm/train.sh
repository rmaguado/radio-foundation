#!/bin/bash

export DHOME="$HOME/projects/radio-foundation"
export OUT="$DHOME/runs"

# NAME NODE GPUS NWORKERS CONFIG

export NAME=$1
export NODE=$2
export GPUS=$3
export WORKERS=$4
export CONFIG=$5
export ZERO=$6

export MEMORY=$(($GPUS*50))GB

if [ ! -z $6 ]; then 
  printf "\n     name | $NAME\n     node | $NODE\nresources | gpu:$GPUS workers:$WORKERS \n\n"
  mkdir -p $OUT/$NAME
  envsubst '$NAME $NODE $GPUS $WORKERS $CONFIG $OUT $ZERO $MEMORY' < $DHOME/scripts/train/vlm/train.template > $OUT/$NAME/train.run
  sbatch $OUT/$NAME/train.run

else
  printf "\nneed NAME NODE GPUS NWORKERS CONFIG ZERO\n\n"

fi
