import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

def compile_parquet_to_memmap(parquet_dir: str, output_dat_path: str, obs_dim: int, act_dim: int, z_dim: int):
    """
    One-time utility to convert partitioned PySpark Parquet files into a flat 
    binary memory-mapped file for zero-RAM streaming.
    """
    print(f"Compiling Parquet files from {parquet_dir} to MemMap...")
    # Read PySpark output (Iteratively in chunks in production to save RAM)
    df = pd.read_parquet(parquet_dir)
    
    # Calculate total feature dimension per timestep
    total_dim = obs_dim + act_dim + z_dim
    num_rows = len(df)
    
    # Create the binary file on the NVMe drive
    fp = np.memmap(output_dat_path, dtype='float32', mode='w+', shape=(num_rows, total_dim))
    
    # Write data directly to disk bypassing heavy RAM allocation
    # (Assuming columns are ordered: [observations, actions, z_classes])
    fp[:] = df.values.astype('float32')
    fp.flush()
    print(f"MemMap compiled successfully: {output_dat_path} | Shape: {fp.shape}")

class SocietalMemMapDataset(Dataset):
    """
    Streams deep historical sequences directly from NVMe to VRAM.
    Consumes essentially 0MB of the 64GB System RAM.
    """
    def __init__(self, dat_path: str, num_rows: int, obs_dim: int, act_dim: int, z_dim: int, seq_len: int):
        self.total_dim = obs_dim + act_dim + z_dim
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Open in 'r' (read-only) mode. The OS kernel handles page caching automatically.
        self.data = np.memmap(dat_path, dtype='float32', mode='r', shape=(num_rows, self.total_dim))
        
    def __len__(self):
        # We can sample any sequence of length `seq_len` up to the end of the dataset
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        # Slice the memory map. This reads instantly from the SSD via OS paging.
        # History window (T)
        history_slice = self.data[idx : idx + self.seq_len]
        # Target step (T + 1) for JEPA EMA target
        next_step = self.data[idx + 1 : idx + self.seq_len + 1]
        
        # Convert to tensors
        hist_tensor = torch.from_numpy(history_slice)
        next_tensor = torch.from_numpy(next_step)
        
        # Unpack dimensions
        obs_h = hist_tensor[:, :self.obs_dim]
        act_h = hist_tensor[:, self.obs_dim : self.obs_dim + self.act_dim]
        z_h = hist_tensor[:, -self.z_dim:]
        
        obs_next = next_tensor[:, :self.obs_dim]
        act_next = next_tensor[:, self.obs_dim : self.obs_dim + self.act_dim]
        
        # Extract the specific action and Z-observation at the end of the sequence
        a_t = act_h[-1]
        z_t = z_h[-1]
        
        return obs_h, act_h, a_t, z_t, obs_next, act_next