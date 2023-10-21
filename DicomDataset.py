import os
import torch
import pydicom
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
"""
# Train images: ImageData/train/0; ImageData/train/1
# Test images: ImageData/test/0; ImageData/test/1
# Classes: "not_operable": 0; "operable":1
#
# set_file_matrix():
# Count total items in sub-folders of root/image_dir:
# Create a list with all items from root/image_dir "(pixel_array, label)"
# Takes label from sub-folder name "0" or "1"
"""
class DicomDataset(Dataset):
    def __init__(self, root, csv, transform=None):
        self.root = root
        self.df = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image_file_1 = pydicom.dcmread(os.path.join(self.root, self.df.iat[index, 0]))
        image_file_2 = pydicom.dcmread(os.path.join(self.root, self.df.iat[index, 1]))
        image_1 = np.array(image_file_1.pixel_array, dtype=np.float32)[np.newaxis]  # Add channel dimension
        image_2 = np.array(image_file_2.pixel_array, dtype=np.float32)[np.newaxis]  # Add channel dimension
        image_1 = torch.from_numpy(image_1)
        image_2 = torch.from_numpy(image_2)
        target = torch.from_numpy(np.array(self.df.iat[index, 2], dtype=np.float32))
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return (image_1, image_2, target)
    
