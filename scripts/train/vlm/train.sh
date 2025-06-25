#!/bin/bash

export DHOME="$HOME/projects/radio-foundation"
export OUT="$DHOME/runs"

# NAME NODE GPUS NWORKERS CONFIG

export NAME=$1
export NODE=$2
export GPUS=$3
export GPUTYPE=$4
export WORKERS=$5
export CONFIG=$6
export ZERO=$7

export MEMORY=$(($GPUS*96))GB

if [ ! -z $7 ]; then 
  printf "\n     name | $NAME\n     node | $NODE\nresources | gpu:$GPUS workers:$WORKERS \n\n"
  mkdir -p $OUT/$NAME
  envsubst '$NAME $NODE $GPUS $GPUTYPE $WORKERS $CONFIG $OUT $ZERO $MEMORY' < $DHOME/scripts/train/vlm/train.template > $OUT/$NAME/train.run
  sbatch $OUT/$NAME/train.run

else
  printf "\nneed NAME NODE GPUS NWORKERS CONFIG ZERO\n\n"

fi
