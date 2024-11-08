#!/bin/bash

export DHOME="$HOME/projects/radio-foundation"
export OUT="$PWD/runs"

if [ ! -e $DHOME/sjob.template ]; then
  echo "sjob.template not found"
  exit 1
fi

# CONF NODES GPUS [W]

export CONF="${1##*/}"
export CONF="${CONF%.*}"
export GPUS=$2
export GPUTYPE=${3:-L40S}

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
  printf "\nJob  conf: $CONF  gpus-per_node: $GPUS $GPUTYPE GPUs\n\n"
  mkdir -p $OUT/$CONF
  envsubst '$CONF $GPUS $GPUTYPE $OUT' < $PWD/sjob.template > $OUT/$CONF/$CONF.run
  sbatch $OUT/$CONF/$CONF.run

else
  printf "\nneed CONF GPUS GPUTYPE\n\n"

fi
