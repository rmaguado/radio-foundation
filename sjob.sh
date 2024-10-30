#!/bin/bash

CONF="${1##*/}"
export CONF="${CONF%.*}"

export GPUS=$2
export GPUTYPE=${3:-A40}

export OUTPUTDIR="$PWD/runs/$CONF"
mkdir $OUTPUTDIR

if [ ! -z $2 ]; then 
  printf "$GPUTYPE:$GPUS conf:$CONF\n"

  envsubst '$CONF $GPUS $GPUTYPE $OUTPUTDIR' < sjob.template > "$OUTPUTDIR/sjob.tmp"
  sbatch "$OUTPUTDIR/sjob.tmp"

else
  printf "CONF nGPUS [GPUTYPE]\n"

fi
