#!/bin/bash

source $HOME/.bashrc
conda activate dino2t24
cd $HOME/projects/radio-foundation
export PYTHONPATH=dinov2


CONF="${1##*/}"
CONF="${CONF%.*}"

NNODES=${2:-1}
GPUS=${3:-2}
GPUTYPE=${4:-'L40S'}

printf "\n\nJob  conf: $CONF  nodes: $NNODES gpus-per_node: $GPUS $GPUTYPE\n\n"

# printf "\n\nCUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}\n\n"
# export NCCL_DEBUG=INFO


if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then

printf "SSH job CUDA_VISIBLE_DEVICES=:$CUDA_VISIBLE_DEVICES\n"

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
mkdir -p outs/${CONF} # log

torchrun --nnodes=1 \
    --nproc_per_node=${GPUS} \
    --rdzv-endpoint="127.0.0.1:$(find_port)" \
        ${PYTHONPATH}/dinov2/train/train.py \
            --config-file="${PWD}/configs/${CONF}.yaml" \
            --output-dir="${PWD}/outs/${CONF}" > "${PWD}/outs/${CONF}/${CONF}.log" 2>&1 &

else

# printf "SLRUM job\n"

if [ -f "${PWD}/configs/${CONF}.yaml" ]; then 

if [ ! -z $GPUTYPE ]; then GRES="--gres gpu:$GPUTYPE:$GPUS"; fi

python ${PYTHONPATH}/dinov2/run/train/train.py \
    --nnodes $NNODES --mem 64 --timeout 14400 --partition 'odap-gpu' \
    --cpus 8 --ngpus $GPUS $GRES \
    --config-file="${PWD}/configs/${CONF}.yaml" \
    --output-dir="${PWD}/outs/${CONF}"
else
  printf "config not found: ${PWD}/configs/${CONF}.yaml\n\n"
fi

fi