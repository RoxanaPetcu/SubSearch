#!/bin/bash
#SBATCH -p gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=16

# This runs your existing training script on a single GPU node
bash train_grpo.sh