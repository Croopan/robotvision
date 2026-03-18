import os
import torch
import numpy as np
from torch.utils.data import IterableDataset
from typing import Optional, Dict, List, Tuple
import pickle
from tqdm import tqdm

from protos import e2e_pb2
from viz_camera_projection import _build_segment_index, get_camera_calibration, decode_image

# The three front-facing cameras from Waymo dataset
FRONT3_CAMERAS = [2, 1, 3] # FRONT_LEFT, FRONT, FRONT_RIGHT

class WaymoVideoClipDataset(IterableDataset):
    """
    Dataset that yields temporally-ordered "video clips" (sequences of frames) 
    from driving segments. This is suitable for Temporal BEV models like BEVStereo
    or Lift-Splat-Shoot that require sequences of multi-view camera inputs.
    """
    def __init__(
        self,
        index_file: str,
        data_dir: str,
        clip_length: int = 3, # number of frames per clip
        stride: int = 1,      # stride between clips (temporal stride)
        min_segment_length: int = 50,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.clip_length = clip_length
        self.stride = stride
        
        # Build or load segment index mapping segments to sorted frames
        self.seg_index = _build_segment_index(index_file, data_dir)
        
        # Keep segments that are long enough to form at least one clip
        self.valid_segments = [
            (seg, frames) 
            for seg, frames in self.seg_index.items() 
            if len(frames) >= min_segment_length and len(frames) >= clip_length
        ]
        
        # Sort for determinism across runs
        self.valid_segments.sort(key=lambda x: x[0])
        
        # Shuffle with seed for dataset variety
        import random
        rng = random.Random(seed)
        rng.shuffle(self.valid_segments)

    def _parse_frame(self, file_handle, start_byte, byte_length):
        file_handle.seek(start_byte)
        protobuf = file_handle.read(byte_length)
        frame = e2e_pb2.E2EDFrame()
        frame.ParseFromString(protobuf)

        cam_images = {}
        for img in frame.frame.images:
            if img.name in FRONT3_CAMERAS:
                # the images are stored as jpeg bytes, we decode them
                img_array = decode_image(img.image)
                cam_images[img.name] = img_array

        calibrations = {}
        for cam_id in FRONT3_CAMERAS:
            try:
                intr, extr, dist, w, h = get_camera_calibration(
                    frame.frame.context.camera_calibrations, cam_id
                )
                calibrations[cam_id] = {
                    "intrinsic": intr,
                    "extrinsic": extr,
                    "dist": dist
                }
            except ValueError:
                pass 

        past = np.stack(
            [
                frame.past_states.pos_x,
                frame.past_states.pos_y,
                frame.past_states.vel_x,
                frame.past_states.vel_y,
                frame.past_states.accel_x,
                frame.past_states.accel_y,
            ],
            axis=-1,
        ).astype(np.float32)

        future = np.stack(
            [
                frame.future_states.pos_x,
                frame.future_states.pos_y,
            ],
            axis=-1,
        ).astype(np.float32)

        return {
            "CAM_IMAGES": cam_images,
            "CALIBRATIONS": calibrations,
            "PAST": past,
            "FUTURE": future,
            "INTENT": frame.intent,
            "NAME": frame.frame.context.name
        }

    def __iter__(self):
        current_file = None
        current_filename = None

        for seg_id, frames in self.valid_segments:
            # Moving window over the segment with self.stride
            for start_idx in range(0, len(frames) - self.clip_length + 1, self.stride):
                clip_frames_meta = frames[start_idx : start_idx + self.clip_length]
                
                clip_data = []
                for frame_idx, filename, start_byte, byte_length in clip_frames_meta:
                    if filename != current_filename:
                        if current_file:
                            current_file.close()
                        current_file = open(os.path.join(self.data_dir, filename), "rb")
                        current_filename = filename
                    
                    frame_data = self._parse_frame(current_file, start_byte, byte_length)
                    clip_data.append(frame_data)
                
                # Stack clip data
                # Typically for BEVStereo we want tensors of shape (T, N_cams, C, H, W)
                
                T = len(clip_data)
                
                clip_images = []
                clip_intrinsics = []
                clip_extrinsics = []
                
                for t in range(T):
                    t_images = []
                    t_intrinsics = []
                    t_extrinsics = []
                    for cam_id in FRONT3_CAMERAS:
                        # Convert to tensor (C, H, W) and normalize roughly, or keep as uint8
                        # We keep as float32 in [0, 1] as an example
                        img_tensor = torch.from_numpy(clip_data[t]["CAM_IMAGES"][cam_id]).permute(2, 0, 1).float() / 255.0
                        t_images.append(img_tensor)
                        
                        intr = torch.from_numpy(clip_data[t]["CALIBRATIONS"][cam_id]["intrinsic"]).float()
                        extr = torch.from_numpy(clip_data[t]["CALIBRATIONS"][cam_id]["extrinsic"]).float()
                        
                        t_intrinsics.append(intr)
                        t_extrinsics.append(extr)
                        
                    clip_images.append(torch.stack(t_images))          # (N_cams, C, H, W)
                    clip_intrinsics.append(torch.stack(t_intrinsics))  # (N_cams, 3, 3)
                    clip_extrinsics.append(torch.stack(t_extrinsics))  # (N_cams, 4, 4)
                
                # yield a single clip dict
                yield {
                    "images": torch.stack(clip_images),         # (T, N_cams, C, H, W)
                    "intrinsics": torch.stack(clip_intrinsics), # (T, N_cams, 3, 3)
                    "extrinsics": torch.stack(clip_extrinsics), # (T, N_cams, 4, 4)
                    "past": torch.from_numpy(clip_data[-1]["PAST"]), # Ground truth from the last frame of the clip
                    "future": torch.from_numpy(clip_data[-1]["FUTURE"]),
                    "intent": clip_data[-1]["INTENT"],
                    "name": clip_data[-1]["NAME"],
                }

        if current_file:
            current_file.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/scratch/gilbreth/svelmuru/waymo_end_to_end_dataset/waymo_open_dataset_end_to_end_camera_v_1_0_0")
    parser.add_argument("--index_file", type=str, default="index_val.pkl")
    parser.add_argument("--clip_length", type=int, default=3, help="Number of frames per clip")
    parser.add_argument("--num_clips", type=int, default=10, help="Number of clips to extract for testing")
    args = parser.parse_args()

    # Instantiate dataset
    dataset = WaymoVideoClipDataset(
        index_file=args.index_file,
        data_dir=args.data_root,
        clip_length=args.clip_length,
        stride=1,
    )

    print(f"Dataset initialized with {len(dataset.valid_segments)} valid driving segments out of {len(dataset.seg_index)}")
    
    # Iterate and grab simply a few clips
    clips_collected = 0
    for clip in tqdm(dataset, total=args.num_clips, desc="Accumulating Video Clips"):
        clips_collected += 1
        
        # Here you would typically save or feed the clip into a BEVStereo dataloader/pipeline.
        # Format mapping:
        # clip["images"] -> Shape: (T, N_cams, 3, 1280, 1920)
        # clip["intrinsics"] -> Shape: (T, N_cams, 3, 3)
        # clip["extrinsics"] -> Shape: (T, N_cams, 4, 4)
        
        if clips_collected == 1:
            print("\nSuccessfully accumulated first sequence clip!")
            print("Clip Data shapes:")
            print(f" - Images:     {clip['images'].shape}")
            print(f" - Intrinsics: {clip['intrinsics'].shape}")
            print(f" - Extrinsics: {clip['extrinsics'].shape}")
            print(f" - Past:       {clip['past'].shape}")
            print(f" - Future:     {clip['future'].shape}")
            print(f" - Intent:     {clip['intent']}")
            print(f" - Name:       {clip['name']}")
        
        if clips_collected >= args.num_clips:
            break

    print(f"\nFinished accumulating {clips_collected} clips. The pipeline is ready to be connected to BEVStereo.")

if __name__ == "__main__":
    main()
