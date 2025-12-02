
import os
import torch
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SDF_dataset(Dataset):
    """Dataset class for loading and serving SDF training samples.
    
    Loads preprocessed samples from numpy files and converts them to PyTorch tensors.
    Combines samples from all objects into a single dataset for efficient batching.
    """

    def __init__(self, data_folder_path: str):
        """Initialize dataset by loading samples from disk.
        
        Loads samples_dict.npy containing SDF values and coordinates for all objects.
        Combines data from all objects into unified tensors for efficient access.
        
        Args:
            data_folder_path: Path to folder containing samples_dict.npy file.
        """
        samples_dict = np.load(
            os.path.join(data_folder_path, 'samples_dict.npy'),
            allow_pickle=True
        ).item()

        self.data = dict()
        for obj_idx in list(samples_dict.keys()):
            for key in samples_dict[obj_idx].keys():
                value = torch.from_numpy(samples_dict[obj_idx][key]).float()
                if len(value.shape) == 1:
                    value = value.view(-1, 1)
                if key not in list(self.data.keys()):
                    self.data[key] = value
                else:
                    self.data[key] = torch.vstack((self.data[key], value))

    # ========== Public API ==========

    def __len__(self) -> int:
        """Get total number of samples in dataset.
        
        Returns:
            Number of SDF samples across all objects.
        """
        return self.data['sdf'].shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample by index.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (latent_class_coords, sdf_value) where:
                - latent_class_coords: Tensor of shape (4,) containing [class_idx, x, y, z]
                - sdf_value: Scalar tensor containing SDF value
        """
        latent_class = self.data['samples_latent_class'][idx, :]
        sdf = self.data['sdf'][idx]
        return latent_class, sdf