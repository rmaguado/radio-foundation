#!/bin/bash

# CONF NODES GPUS [W]

CONF="${1##*/}"
export CONF="${CONF%.*}"

export NODES=$2
export GPUS=$3
export W=${4:-2}
export PORT=$(find_port)


if [ ! -z $3 ]; then 
  printf "\n\nJob  conf: $CONF  nodes: $NODES gpus-per_node: $GPUS compute-cuda-0$W\n\n"

  envsubst '$CONF $NODES $GPUS $W $PORT' < sjob.template > sjob.tmp
  sbatch sjob.tmp

else
  printf "\n\nneed CONF NODES GPUS VISIBLE [W]\n\n"

fi
