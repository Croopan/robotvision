#!/bin/bash
#SBATCH --job-name=waymo_bev_gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --account=csso

# Load modules (edit for your cluster)
module load cuda/12.1
module load conda

# Activate your environment
conda activate waymo_env

# ── Paths (hardcoded — BASH_SOURCE doesn't work under Slurm) ──
SCRIPT_DIR=/home/svelmuru/robotvision/src/camera-based-e2e/Point_cloud
PARENT_DIR=/home/svelmuru/robotvision/src/camera-based-e2e
DATA_DIR=/scratch/gilbreth/svelmuru/waymo_end_to_end_dataset/waymo_open_dataset_end_to_end_camera_v_1_0_0

# Parent dir has protos/ and index_*.pkl
export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH}"

echo "Processing all frames in a single GPU job"

python "${SCRIPT_DIR}/create_bev.py" \
    --data_dir $DATA_DIR \
    --index_dir "${PARENT_DIR}" \
    --split train
