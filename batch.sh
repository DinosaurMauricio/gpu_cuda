#!/bin/bash
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=test
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err

if [ "$#" -eq 2 ]; then
    srun ./bin/gpu_transpose "$1" "$2"
    exit 1
else 
    echo "Usage: $0 arg1 arg2"
    exit 1
fi