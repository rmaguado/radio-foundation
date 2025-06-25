#!/bin/bash

export DHOME="$HOME/projects/radio-foundation"
export OUT="$DHOME/runs"

# NAME NODE CONFIG CHECKPOINT 

export NAME=$1
export NODE=$2
export PATH_TO_RUN=$3
export CHECKPOINT=$4
export GPUS=1
export WORKERS=8

export MEMORY=$(($GPUS*96))GB

if [ ! -z $4 ]; then 
  printf "\n     name | $NAME\n     node | $NODE\nresources | gpu:$GPUS workers:$WORKERS \n\n"
  mkdir -p $OUT/$NAME
  envsubst '$NAME $NODE $GPUS $WORKERS $OUT $PATH_TO_RUN $CHECKPOINT $MEMORY' < $DHOME/scripts/generate/generate.template > $OUT/$NAME/train.run
  sbatch $OUT/$NAME/train.run

else
  printf "\nneed NAME NODE PATH_TO_RUN CHECKPOINT\n\n"

fi
