#!/bin/bash

export DHOME="$HOME/projects/radio-foundation"

export NAME=$1
export NODE=$2
export GPUS=$3
export GPUTYPE=$4
export WORKERS=$5
export MEM=$6

export OUT="$DHOME/runs/$NAME"


mkdir -p $OUT

envsubst '$DHOME $OUT $NAME $NODE $GPUS $GPUTYPE $WORKERS  $MEM' < $DHOME/scripts/tests/sjob.template > $OUT/test.run

sbatch $OUT/test.run