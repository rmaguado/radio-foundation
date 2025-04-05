#!/bin/bash

export DHOME="$HOME/projects/radio-foundation"
export OUT="$DHOME/runs"

# NAME NODE GPUS NWORKERS CONFIG

export NAME=$1
export NODE=$2
export CONFIG=$3
export CHECKPOINT=$4
export GPUS=1
export WORKERS=4

export MEMORY=$(($GPUS*96))GB

if [ ! -z $4 ]; then 
  printf "\n     name | $NAME\n     node | $NODE\nresources | gpu:$GPUS workers:$WORKERS \n\n"
  mkdir -p $OUT/$NAME
  envsubst '$NAME $NODE $GPUS $WORKERS $CONFIG $OUT $CHECKPOINT $MEMORY' < $DHOME/scripts/generate/generate.template > $OUT/$NAME/train.run
  sbatch $OUT/$NAME/train.run

else
  printf "\nneed NAME NODE CONFIG CHECKPOINT\n\n"

fi
