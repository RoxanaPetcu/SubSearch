#!/bin/bash
#SBATCH -p gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16

bash eval_grpo.sh