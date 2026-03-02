import torch
from torch.utils.data import IterableDataset
from protos import e2e_pb2
import torchvision
import pickle
import struct
import os
import numpy as np
from PIL import Image
from io import BytesIO 
import cv2 
from typing import Optional
import random
import tqdm
import open3d as o3d
from point_cloud import create_multi_view_point_cloud
from depthLoss import DepthLoss

devices = ['cuda:0', 'cuda:1']

random.seed(42) # Deterministic

class WaymoE2E(IterableDataset): 
    def __init__(
        self,
        indexFile = 'index.pkl',
        data_dir='./dataset',
        images = True,
        n_items: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.images = images
        self.data_dir = data_dir
        self.seed = seed

        self.filename = ""
        self.file = None

        with open(indexFile, 'rb') as f:
            # NOTE: test does not have reference trajectories
            # We train on train and validate on val set
            self.indexes = pickle.load(f)

        # TODO: Determine how to sample specific subsets of the data that we care about.
        if n_items is not None and n_items < len(self.indexes):
            total = len(self.indexes)
            # pick a deterministic contiguous block when a seed is provided
            rng = random.Random(seed) if seed is not None else random
            start = rng.randint(0, total - n_items)
            self.indexes = self.indexes[start : start + n_items]



    def decode_img(self, img):
        if not self.images:
            return np.array([])
        
        img_tensor = torch.from_numpy(np.frombuffer(img, dtype=np.uint8).copy()) 
        gpu_tensors_list = torchvision.io.decode_jpeg(
            img_tensor, 
            mode=torchvision.io.ImageReadMode.UNCHANGED,
            device= 'cpu', #['cuda:0', 'cuda:1'][torch.utils.data.get_worker_info().id%2]
        )
        # img_array = np.frombuffer(img, np.uint8)
        return gpu_tensors_list
    
    def __len__(self):
        return len(self.indexes)
    
    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            start, step = 0, 1
        else:
            start, step = worker.id, worker.num_workers

        for idx in range(start, len(self.indexes), step):
            frame = e2e_pb2.E2EDFrame()  # type: ignore
            filename, start_byte, byte_length = self.indexes[idx]

            if self.filename != filename:
                if self.file:
                    self.file.close()
                    del self.file
                self.file = open(os.path.join(self.data_dir, filename), 'rb')
                self.filename = filename

            self.file.seek(start_byte) # type: ignore
            protobuf = self.file.read(byte_length) # type: ignore
            frame.ParseFromString(protobuf)

            past = np.stack([frame.past_states.pos_x, frame.past_states.pos_y, frame.past_states.vel_x, frame.past_states.vel_y, frame.past_states.accel_x, frame.past_states.accel_y], axis=-1)
    

            future = np.stack([frame.future_states.pos_x, frame.future_states.pos_y], axis=-1)

            past = np.array(past, dtype=np.float32) # ensure consistent dtype
            future = np.array(future, dtype=np.float32)

            # For submission to waymo evaluation server
            name = frame.frame.context.name

            intrinsics_vec = np.zeros(6, dtype=np.float32)
            
            # Find FRONT camera (Enum 1) in the context
            for calib in frame.frame.context.camera_calibrations:
                if calib.name == 1: # 1 = FRONT
                    intrinsics_vec[0] = calib.intrinsic[0] # fx
                    intrinsics_vec[1] = calib.intrinsic[1] # fy
                    intrinsics_vec[2] = calib.intrinsic[2] # cx
                    intrinsics_vec[3] = calib.intrinsic[3] # cy
                    intrinsics_vec[4] = calib.width
                    intrinsics_vec[5] = calib.height
                    break
            
            decoded_images = [self.decode_img(img.image) for img in frame.frame.images]

            # Extract per-camera calibration data for point cloud fusion
            camera_names = []
            all_intrinsics = []  # [fx, fy, cx, cy, width, height, k1, k2, p1, p2, k3] per camera
            all_extrinsics = []  # 4x4 matrix per camera
            for img_proto in frame.frame.images:
                cam_name = img_proto.name
                camera_names.append(cam_name)
                for c in frame.frame.context.camera_calibrations:
                    if c.name == cam_name:
                        all_intrinsics.append([
                            c.intrinsic[0], c.intrinsic[1],  # fx, fy
                            c.intrinsic[2], c.intrinsic[3],  # cx, cy
                            c.width, c.height,
                            c.intrinsic[4], c.intrinsic[5],  # k1, k2
                            c.intrinsic[6], c.intrinsic[7],  # p1, p2
                            c.intrinsic[8],                   # k3
                        ])
                        all_extrinsics.append(
                            np.array(c.extrinsic.transform, dtype=np.float64).reshape(4, 4)
                        )
                        break

            yield {
                'PAST': past, 
                'FUTURE': future, 
                'IMAGES': decoded_images, 
                'INTRINSICS': intrinsics_vec,
                'NAME': frame.frame.context.name,
                'CAMERA_NAMES': np.array(camera_names, dtype=np.int32),
                'ALL_INTRINSICS': np.array(all_intrinsics, dtype=np.float64),
                'ALL_EXTRINSICS': np.array(all_extrinsics, dtype=np.float64),
            }

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    import time
    from tqdm import tqdm
    # NOTE: Replace with your path
    DATA_DIR = '/scratch/gilbreth/svelmuru/waymo_end_to_end_dataset/waymo_open_dataset_end_to_end_camera_v_1_0_0/'
    BATCH_SIZE = 32
    dataset = WaymoE2E(indexFile="index_train.pkl", data_dir = DATA_DIR, images=True, n_items= 5)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        num_workers=16,
    )
    
    def main():
        device = torch.device("cuda")
        depth_model = DepthLoss(device)
        output_dir = "visualizations"
        os.makedirs(output_dir, exist_ok=True)

        for batch_idx, batch_of_frames in enumerate(tqdm(loader)):
            all_camera_images = batch_of_frames["IMAGES"]  # 5 X (B, 3, H, W)
            batch_size = all_camera_images[0].shape[0] # B

            for i in range(batch_size):
                frame_id = batch_idx * BATCH_SIZE + i

                # Calibration data 
                cam_names = batch_of_frames['CAMERA_NAMES'][i]    # (N_cams,)
                cam_intrinsics = batch_of_frames['ALL_INTRINSICS'][i]  # (N_cams, 6)
                cam_extrinsics = batch_of_frames['ALL_EXTRINSICS'][i]  # (N_cams, 4, 4)
                num_cameras = cam_names.shape[0]

                rgb_list = []
                depth_list = []

                for cam_idx in range(num_cameras):
                    cam_enum = cam_names[cam_idx].item()

                    # Get the specific image
                    img_tensor = all_camera_images[cam_idx][i]  # (3, H, W)
                    img_gpu = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

                    # Run depth estimation
                    pred_depth = depth_model.get_depth(img_gpu)  # (1, H, W)
                    depth_np = pred_depth[0].cpu().numpy()

                    # Convert image to uint8 numpy
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy().copy()
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)

                    rgb_list.append(img_np)
                    depth_list.append(depth_np)

                    # Save per-camera image
                    img_filename = f"frame_{frame_id:04d}_cam_{cam_enum}.png"
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_dir, img_filename), img_bgr)

                # Fuse all cameras into one point cloud in vehicle frame
                fused_pcd = create_multi_view_point_cloud(
                    rgb_list, depth_list,
                    cam_intrinsics.numpy(), cam_extrinsics.numpy()
                )

                # Save fused point cloud
                pcd_filename = f"frame_{frame_id:04d}_fused.pcd"
                o3d.io.write_point_cloud(
                    os.path.join(output_dir, pcd_filename), fused_pcd
                )
                print(f"  Frame {frame_id}: {len(fused_pcd.points)} points "
                      f"from {num_cameras} cameras")

    import cProfile
    main()