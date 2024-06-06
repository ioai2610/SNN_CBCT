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
        self.root = root # directorio base
        self.df = pd.read_csv(csv) # lectura del archivo csv
        self.transform = transform # aplicar una transformacion

    def __len__(self):
        return self.df.shape[0] # longitud del archivo csv

    def __getitem__(self, index):
        # lectura de los primeros dos archivos (columna 1 y columna 2)
        image_file_1 = pydicom.dcmread(os.path.join(self.root, self.df.iat[index, 0])) 
        image_file_2 = pydicom.dcmread(os.path.join(self.root, self.df.iat[index, 1]))
        # convertir a arreglo numpy con instrucción pixel_array para leer archivos DICOM y añadimos una dimension
        image_1 = np.array(image_file_1.pixel_array, dtype=np.float32)[np.newaxis]  # Add channel dimension | debe ser float32 para poder operar en tensor
        image_2 = np.array(image_file_2.pixel_array, dtype=np.float32)[np.newaxis]  # Add channel dimension | debe ser float32 para poder operar en tensor
        # pasamos de numpy a un tensor en torch
        image_1 = torch.from_numpy(image_1)
        image_2 = torch.from_numpy(image_2)
        # hacemos lo mismo para el objetivo (puntuacion)
        target = torch.from_numpy(np.array(self.df.iat[index, 2], dtype=np.float32)) # | debe ser float32 para poder operar en tensor
        # si se define una transformación, esta se aplica
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        # obtenemos ambas imagenes y su puntuacion
        return (image_1, image_2, target)
        # try:
        #     # Lectura de los primeros dos archivos (columna 1 y columna 2)
        #     image_file_1 = pydicom.dcmread(os.path.join(self.root, self.df.iat[index, 0])) 
        #     image_file_2 = pydicom.dcmread(os.path.join(self.root, self.df.iat[index, 1]))
            
        #     # Convertir a arreglo numpy con instrucción pixel_array para leer archivos DICOM y añadimos una dimensión
        #     image_1 = np.array(image_file_1.pixel_array, dtype=np.float32)[np.newaxis]  # Add channel dimension
        #     image_2 = np.array(image_file_2.pixel_array, dtype=np.float32)[np.newaxis]  # Add channel dimension
            
        #     # Pasamos de numpy a un tensor en torch
        #     image_1 = torch.from_numpy(image_1)
        #     image_2 = torch.from_numpy(image_2)
            
        #     # Hacemos lo mismo para el objetivo (puntuación)
        #     target = torch.from_numpy(np.array(self.df.iat[index, 2], dtype=np.float32))
            
        #     # Si se define una transformación, esta se aplica
        #     if self.transform:
        #         image_1 = self.transform(image_1)
        #         image_2 = self.transform(image_2)

        #     # Obtenemos ambas imágenes y su puntuación
        #     return (image_1, image_2, target)
        
        # except AttributeError as e:
        #     print(f"Error reading DICOM file at index {index}: {e}")
        #     return None
    
