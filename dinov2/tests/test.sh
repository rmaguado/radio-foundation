#!/bin/bash

#SBATCH --job-name=dinov2tests
#SBATCH --output=runs/dinov2tests/%j.out
#SBATCH --nodes=1
#SBATCH --nodelist=compute-cuda-02
#SBATCH --cpus-per-task=1

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1

#SBATCH --mem=64GB
#SBATCH --partition=odap-gpu
#SBATCH --signal=USR2@120
#SBATCH --time=14400

source $HOME/.conda_rc
conda activate radio
cd $HOME/projects/radio-foundation
export PYTHONPATH=$PWD

pytest dinov2/tests/test_train.py --tb=long --log-cli-level=DEBUG