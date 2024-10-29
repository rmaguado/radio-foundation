#!/bin/bash

# CONF NODES GPUS [W]

CONF="${1##*/}"
export CONF="${CONF%.*}"

export NODES=$2
export GPUS=$3
export W=${4:-2}


if [ ! -z $3 ]; then 
  printf "Job  conf: $CONF  nodes: $NODES gpus-per_node: $GPUS compute-cuda-0$W\n"

  envsubst '$CONF $NODES $GPUS $W $PORT' < sjob.template > sjob.tmp
  sbatch sjob.tmp

else
  printf "Need \$CONF \$NODES \$GPUS \$VISIBLE [\$W]\n"

fi
