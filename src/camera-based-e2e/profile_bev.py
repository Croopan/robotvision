import cProfile
import pstats
import io
import torch
import numpy as np
import pickle
import os
import time

from protos import e2e_pb2
from Point_cloud.create_bev import load_depth_model
from Point_cloud.point_cloud_gpu import create_bev_from_frame_gpu

DATA_DIR = "/scratch/gilbreth/svelmuru/waymo_end_to_end_dataset/waymo_open_dataset_end_to_end_camera_v_1_0_0"
INDEX_FILE = os.path.join(DATA_DIR, "..", "index_train.pkl")

with open(INDEX_FILE, "rb") as f:
    indexes = pickle.load(f)

device = torch.device("cuda")
model, processor = load_depth_model(device)

open_file = None
open_filename = ""

def process_frame(idx):
    global open_file, open_filename
    filename, start_byte, byte_length = indexes[idx]
    if open_filename != filename:
        if open_file: open_file.close()
        open_file = open(os.path.join(DATA_DIR, filename), "rb")
        open_filename = filename
    open_file.seek(start_byte)
    protobuf = open_file.read(byte_length)
    frame = e2e_pb2.E2EDFrame()
    frame.ParseFromString(protobuf)
    
    # Warmup torch
    if idx == 0:
        create_bev_from_frame_gpu(frame, model, processor, device)
        
    torch.cuda.synchronize()
    start = time.perf_counter()
    bev = create_bev_from_frame_gpu(frame, model, processor, device)
    torch.cuda.synchronize()
    return time.perf_counter() - start

# Warmup
process_frame(0)

# Profile
pr = cProfile.Profile()
pr.enable()
times = []
for i in range(1, 11):
    times.append(process_frame(i))
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats(20)

print(f"\nAverage time over 10 frames: {np.mean(times)*1000:.2f} ms")
print("\n--- Profiler Output ---")
print(s.getvalue())
