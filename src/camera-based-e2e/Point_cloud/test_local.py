import sys
import os
from pathlib import Path

PARENT_DIR = "/home/svelmuru/robotvision/src/camera-based-e2e"
sys.path.append(PARENT_DIR)

from protos import e2e_pb2
from loader import WaymoE2E, collate_with_images
from torch.utils.data import DataLoader
from Point_cloud.create_bev import load_depth_model
from Point_cloud.point_cloud_gpu import create_bev_from_frame_gpu
import torch

device = torch.device('cuda')
depth_model, depth_processor = load_depth_model(device)

DATA_DIR = "/scratch/gilbreth/svelmuru/waymo_end_to_end_dataset/waymo_open_dataset_end_to_end_camera_v_1_0_0"
index_file = os.path.join(PARENT_DIR, "index_train.pkl")

dataset = WaymoE2E(indexFile=index_file, data_dir=DATA_DIR, n_items=1)
loader = DataLoader(
    dataset,
    batch_size=1,
    collate_fn=collate_with_images,
)

for batch in loader:
    try:
        protobuf = batch['PROTOBUF'][0].tobytes()
        frame = e2e_pb2.E2EDFrame()
        frame.ParseFromString(protobuf)

        bev = create_bev_from_frame_gpu(
            frame,
            depth_model=depth_model,
            depth_processor=depth_processor,
            device=device
        )
        print("BEV Shape:", bev.shape)
    except Exception as e:
        import traceback
        traceback.print_exc()
