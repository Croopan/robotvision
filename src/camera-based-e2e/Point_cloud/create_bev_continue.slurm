#!/bin/bash
#SBATCH --job-name=waymo_bev_cont
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --account=csso
#SBATCH --array=0-12

# Load modules (edit for your cluster)
module load cuda/12.1
module load conda

# Activate your environment
conda activate waymo_env

DATA_DIR=/scratch/gilbreth/svelmuru/waymo_end_to_end_dataset/waymo_open_dataset_end_to_end_camera_v_1_0_0

# ── Array job: split frames across SLURM tasks ──
TOTAL_TRAIN=415663
NUM_TASKS=13  # must match --array range (0-12)
CHUNK=$(( (TOTAL_TRAIN + NUM_TASKS - 1) / NUM_TASKS ))  # ceiling division (~31,975 frames each)
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

echo "CONTINUATION job: Array task ${SLURM_ARRAY_TASK_ID}/${NUM_TASKS}: processing frames ${START}..${START}+${CHUNK}"
echo "Already-processed .npy files will be skipped automatically."

python create_bev.py \
    --data_dir $DATA_DIR \
    --split train \
    --start_idx $START \
    --n_items $CHUNK
