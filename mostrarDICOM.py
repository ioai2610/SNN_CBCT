import pydicom
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ruta al archivo DICOM
# dicom_file_path = r"C:\Users\iorla\OneDrive\Documentos\Simulaciones Tesis\ImagenesRed\Train_V2_DICOM\FDK_alta_hamming_img.dcm"

# # Leer el archivo DICOM
# dicom_dataset = pydicom.dcmread(dicom_file_path)

# # Obtener la imagen del dataset DICOM
# dicom_image = dicom_dataset.pixel_array
# dicom_image_2 = np.array(dicom_image, dtype=np.float32)[np.newaxis]
# dicom_image_3 = torch.from_numpy(dicom_image_2)

# # Mostrar la imagen usando matplotlib
# plt.imshow(dicom_image, cmap=plt.cm.gray)
# plt.title('Imagen DICOM')
# plt.axis('off')
# plt.show()

print(torch.cuda.is_available())  # Esto debería imprimir True si CUDA está disponible
print(torch.__version__)