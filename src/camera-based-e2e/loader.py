import torch
from torch.utils.data import IterableDataset
from protos import e2e_pb2
import pickle
import os
import numpy as np
from typing import Optional
import random

devices = ['cuda:0', 'cuda:1']

random.seed(42) # Deterministic

class WaymoE2E(IterableDataset): 
    def __init__(
        self,
        indexFile = 'index.pkl',
        data_dir='./dataset',
        n_items: Optional[int] = None,
        seed: Optional[int] = None,
        bev_dir: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.seed = seed
        self.bev_dir = bev_dir

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

    def global_rank(self) -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        
        return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))

    def __len__(self):
        _, world_size = self.global_rank()
        return len(self.indexes) // world_size
    
    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            start, step = self.global_rank()
        else:
            rank, world_size = self.global_rank()
            global_worker_id = rank * worker.num_workers + worker.id
            global_num_workers = world_size * worker.num_workers
            start, step = global_worker_id, global_num_workers

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

            # Yield JPEG images as torch uint8 tensors so that PyTorch
            # DataLoader transfers them via shared memory instead of
            # pickle serialization — critical for multi-worker / DDP perf.
            jpeg_tensors = [
                torch.from_numpy(np.frombuffer(img.image, dtype=np.uint8).copy())
                for img in frame.frame.images
            ]

            sample = {
                'PAST': past, 
                'FUTURE': future, 
                'IMAGES_JPEG': jpeg_tensors, 
                'INTENT': frame.intent, 
                'NAME': name,
                'PROTOBUF': np.frombuffer(protobuf, dtype=np.uint8).copy()
            }

            # Load pre-computed BEV if available
            if self.bev_dir is not None:
                bev_path = os.path.join(self.bev_dir, f'bev_{idx:07d}.npy')
                if os.path.exists(bev_path):
                    sample['BEV'] = np.load(bev_path).astype(np.float32)
                else:
                    sample['BEV'] = np.zeros((4, 200, 200), dtype=np.float32)

            yield sample


def collate_with_images(batch):
    """Collate that keeps IMAGES_JPEG as a list-of-lists (variable-size JPEG
    bytes cannot be stacked) and delegates everything else to default_collate.
    BEV arrays (if present) are regular numpy arrays and collate normally."""
    from torch.utils.data.dataloader import default_collate
    images = [sample.pop('IMAGES_JPEG') for sample in batch]
    protobufs = [sample.pop('PROTOBUF') for sample in batch] if 'PROTOBUF' in batch[0] else None
    
    collated = default_collate(batch)
    collated['IMAGES_JPEG'] = images  # list[list[Tensor]], one inner list per sample
    
    if protobufs is not None:
        # We also keep PROTOBUFs as a list of numpy arrays to avoid collating them
        collated['PROTOBUF'] = protobufs
        
    return collated


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    import time
    from tqdm import tqdm
    # NOTE: Replace with your path
    DATA_DIR = '/scratch/gilbreth/svelmuru/waymo_end_to_end_dataset/waymo_open_dataset_end_to_end_camera_v_1_0_0'
    BATCH_SIZE = 256
    dataset = WaymoE2E(indexFile="index_train.pkl", data_dir = DATA_DIR)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        num_workers=0,
        collate_fn=collate_with_images,
        pin_memory=True, # causes error
    )
    # next(iter(loader))
    
    def main():
        # start = time.time()
        for batch_of_frames in tqdm(loader):
            # print(batch_of_frames["INTENT"])
            # print(batch_of_frames.keys(), [b.shape for b in batch_of_frames.values() if isinstance(b, torch.Tensor)])
            pass
        # print("Total Time:", time.time()-start)
    
    import cProfile
    main()
