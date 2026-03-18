#!/bin/bash
source ~/.bashrc
module load conda
conda activate waymo_env
python test_local.py
