#!/bin/bash
#SBATCH --job-name=waymo_e2e
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=csso

# Load modules (edit for your cluster)
module load cuda/12.1
module load conda

# Activate your environment
conda activate waymo_env

# Run training
python loader.py