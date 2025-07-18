#!/bin/bash

export DHOME="$HOME/projects/radio-foundation"

export NAME=$1
export TESTFILE=$2
export NODE=$3
export GPUS=$4
export GPUTYPE=$5
export WORKERS=$6
export MEM=$7

export OUT="$DHOME/runs/$NAME"


mkdir -p $OUT

envsubst '$DHOME $OUT $NAME $TESTFILE $NODE $GPUS $GPUTYPE $WORKERS  $MEM' < $DHOME/scripts/tests/sjob.template > $OUT/test.run

sbatch $OUT/test.run