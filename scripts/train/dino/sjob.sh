#!/bin/bash

export DHOME="$HOME/projects/radio-foundation"

export NAME=$1
export NODE=$2
export CONF=$3
export GPUS=$4
export GPUTYPE=$5
export WORKERS=$6

export MEM=$(($GPUS * 110))GB

export OUT="$DHOME/runs/$NAME"


mkdir -p $OUT

envsubst '$DHOME $OUT $NAME $CONF $GPUS $GPUTYPE $WORKERS $NODE $MEM' < $DHOME/scripts/train/vision/sjob.template > $OUT/train.run

sbatch $OUT/train.run