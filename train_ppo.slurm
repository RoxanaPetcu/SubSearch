#!/bin/bash
#SBATCH -p gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/trainer_%j.log

# This runs your existing training script on a GPU node
bash train_ppo.sh

