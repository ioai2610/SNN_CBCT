import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_loss_and_metrics(train_loss, test_loss, mae, mse, rmse, epochs): # r2,
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(14, 7))

    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, train_loss, label='Train Loss')
    plt.plot(epochs_range, test_loss, label='Test Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, mae, label='MAE')
    plt.legend(loc='upper right')
    plt.title('Mean Absolute Error')

    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, mse, label='MSE')
    plt.legend(loc='upper right')
    plt.title('Mean Squared Error')

    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, rmse, label='RMSE')
    plt.legend(loc='upper right')
    plt.title('Root Mean Squared Error')

    # plt.subplot(2, 3, 5)
    # plt.plot(epochs_range, r2, label='R^2')
    # plt.legend(loc='upper right')
    # plt.title('R^2 Score')

    plt.tight_layout()
    plt.show()


def plot_loss_and_accuracy_in_one(train_losses, accuracies, epochs): #(train_losses, test_losses, accuracies, epochs)

    epochs = np.arange(1,epochs+1)
    # Crear una figura y un solo eje para pérdida y accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Gráfico de pérdida de entrenamiento y prueba
    ax1.plot(epochs, train_losses, label='Pérdida en entrenamiento', marker='o')
    #ax1.plot(epochs, test_losses, label='Pérdida en prueba', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Pérdida')
    ax1.legend()

    # Crear un segundo eje para la precisión (accuracy)
    ax2 = ax1.twinx()
    ax2.plot(epochs, accuracies, label='Precisión', marker='s', color='green')
    ax2.set_ylabel('Precisión (%)')
    ax2.legend(loc='lower right')

    # Mostrar la gráfica
    plt.title('Pérdida y precisión vs. Epoch')
    plt.show()


# Después de entrenar tu red neuronal, obtén las listas de pérdida y precisión
train_losses = [0.1, 0.08, 0.06, 0.04, 0.03]
test_losses = [0.12, 0.1, 0.08, 0.07, 0.06]
accuracies = [95, 96, 97, 97.5, 98]

# Llama a la función para graficar
#plot_loss_and_accuracy_in_one(train_losses, test_losses, accuracies)

def imshow(img_01,img_02, text=None, should_save=False):
    # npimg_1 = img_01.cpu()
    # npimg_1 = npimg_1.numpy()
    # npimg_2 = img_02.cpu()
    # npimg_2 = npimg_2.numpy()
    # if text:
    #     plt.text(
    #         75,
    #         8,
    #         text,
    #         style="italic",
    #         fontweight="bold",
    #         bbox={"facecolor": "black", "alpha": 0.8, "pad": 10},
    #     )
    # plt.subplot(1,2,1)
    # plt.axis("off")
    # plt.imshow(np.transpose(npimg_1[1],(1,2,0)), cmap='gray') 
    # plt.subplot(1,2,2)
    # plt.axis("off")
    # plt.imshow(np.transpose(npimg_2[1],(1,2,0)), cmap='gray') # np.transpose(npimg, (2, 3, 1, 0))
    # plt.show()
    npimg_1 = img_01.cpu().numpy()
    npimg_2 = img_02.cpu().numpy()
    
    # if text:
    #     plt.text(
    #         0.5,  # Posición x del texto (centro)
    #         0.01,  # Posición y del texto (cerca del borde inferior)
    #         text,
    #         style="italic",
    #         fontweight="bold",
    #         bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    #         transform=plt.gcf().transFigure,  # Transformación para las coordenadas de la figura
    #         horizontalalignment='center',  # Alineación horizontal al centro
    #     )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(text)
    # Configurar ejes y mostrar imágenes
    ax1.axis("off")
    ax2.axis("off")
    ax1.imshow(np.transpose(npimg_1[1], (1, 2, 0)), cmap='gray')
    ax2.imshow(np.transpose(npimg_2[1], (1, 2, 0)), cmap='gray')

    # Agregar la barra de color
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    #norm = plt.Normalize(vmin=0, vmax=11)
    sm = plt.cm.ScalarMappable(cmap='gray')
    sm.set_array([])
    fig.colorbar(sm, cax=cax)

    plt.show()

