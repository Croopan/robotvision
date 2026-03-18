#!/bin/bash
source ~/.bashrc
module load conda
conda activate waymo_env
export PYTHONPATH="/home/svelmuru/robotvision/src/camera-based-e2e:${PYTHONPATH}"
python create_bev.py --data_dir /scratch/gilbreth/svelmuru/waymo_end_to_end_dataset/waymo_open_dataset_end_to_end_camera_v_1_0_0 --n_items 1 --split train
