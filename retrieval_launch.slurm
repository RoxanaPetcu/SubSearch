#!/bin/bash
#SBATCH -p gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --time=120:00:00
#SBATCH --output=logs/retriever_%j.log

# This runs your existing script on a GPU node
bash retrieval_launch.sh