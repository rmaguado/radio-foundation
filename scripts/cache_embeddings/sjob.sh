#!/bin/bash

export DHOME="$HOME/projects/radio-foundation"
export OUT="$DHOME/evaluation/temp"

if [ ! -e $DHOME/sjob.template ]; then
  echo "sjob.template not found"
  exit 1
fi

# NODES GPUS WORKERS [W]

export GPUS=$1
export GPUTYPE=${2:-L40S}
export WORKERS=$3

if [ "$GPUTYPE" != "A40" -a "$GPUTYPE" != "L40S" ]; then
  echo "$GPUTYPE not available"
  exit 1
fi

if [[ "$GPUTYPE" == "A40" ]] && [[ "$GPUS" -gt "2" ]]; then
  export GPUS=2
  echo "max $GPUTYPE changed to 2"
fi
if [[ "$GPUTYPE" == "L40S" ]] && [[ "$GPUS" -gt "4" ]]; then
  export GPUS=4
  echo "max $GPUTYPE changed to 4"
fi


if [ ! -z $2 ]; then 
  printf "\nJob  gpus-per_node: $GPUS $GPUTYPE GPUs   Workers per GPU: $WORKERS\n\n"
  mkdir -p $OUT
  envsubst '$GPUS $GPUTYPE $OUT $WORKERS' < $PWD/sjob.template > $OUT/cache.run
  sbatch $OUT/cache.run

else
  printf "\nneed GPUS GPUTYPE WORKERS\n\n"

fi
