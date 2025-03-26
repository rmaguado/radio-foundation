#!/bin/bash

export DHOME="$HOME/projects/radio-foundation"
export OUT="$DHOME/mllm/runs"

if [ ! -e $DHOME/sjob.template ]; then
  echo "sjob.template not found"
  exit 1
fi

# NODES GPUS WORKERS NAME [W]

export GPUS=$1
export GPUTYPE=${2:-L40S}
export WORKERS=$3
export NAME=$4

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
  printf "\nJob $NAME | gpus-per_node: $GPUS $GPUTYPE GPUs | Workers per GPU: $WORKERS\n\n"
  mkdir -p $OUT/$NAME
  envsubst '$NAME $GPUS $GPUTYPE $OUT $WORKERS' < $DHOME/mllm/train.template > $OUT/$NAME/train.run
  sbatch $OUT/$NAME/train.run

else
  printf "\nneed GPUS GPUTYPE WORKERS\n\n"

fi
