#!/bin/bash
source ~/.bashrc
module load conda
conda activate waymo_env
python profile_bev.py
