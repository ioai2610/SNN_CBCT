import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from utils import plot_loss_and_accuracy_in_one, plot_loss_and_metrics
from torch.optim.lr_scheduler import StepLR
#from SiameseNetworkPytorch import SiameseNetwork, train, test, testShow
from SiameseNetworkPytorch_V2 import SiameseNetwork, train, test, testShow
from DicomDataset import DicomDataset
import numpy as np

# definimos las rutas
train_csv_path = "./Train_Irving_Final_Version2.csv"
test_csv_path = "./Test_Irving_Final_Version2.csv"
# train_csv_path = "./training_data.csv"
# test_csv_path = "./test_data.csv"
root = "./"


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    # tamaño del batch de entrenamiento, default = 64
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)') 
    # tamaño del batch de test, default = 64
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)') 
    # cantidad de epochs, default = 6
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)') 
    # parametro de aprendizaje, default = 1.0
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)') 
    # pasos del parametro de aprendizaje, default = 0.7
    parser.add_argument('--gamma', type=float, default=0.01, metavar='M',
                        help='Learning rate step gamma (default: 0.7)') 
    # por si se desea no usar CUDA
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # para GPU de macOS
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    # semilla para resultados reproducibles
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)') 
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    # guardar modelo
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args() # lectura de argumentos
    
    # uso de CUDA si se encuentra disponible en el equipo
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed) # definición de semilla para reproducibilidad 

    # if use_cuda: # usar CUDA si se encuuentra disponible, de otro modo se usa MPS o el CPU
    #     device = torch.device("cuda")
    # elif use_mps:
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")

    device = torch.device("cuda")

    train_kwargs = {'batch_size': args.batch_size} # argumentos del entrenamiento
    test_kwargs = {'batch_size': args.test_batch_size} # argumentos del test
    if use_cuda: # parámetros si se usa CUDA
        cuda_kwargs = {'num_workers': 4, # si se aumenta el número de subprocesos, puede alentarse la PC
                       'pin_memory': True, # anclar los tensores a la RAM para mejor transferencia de datos a la GPU
                       'shuffle': True} # mezclar o barajar los datos para romper un sesgo inherente en el entrenamiento y garantizar el modelo
        # actualizamos los parametros 
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dataset = DicomDataset(root, train_csv_path) # , transforms.Compose([transforms.Resize(105,antialias=True)])
    test_dataset = DicomDataset(root, test_csv_path) # , transforms.Compose([transforms.Resize(255, antialias=True)])
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = SiameseNetwork().to(device) # aplicacion de la red siamesa y almacenar en "model"
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr) # definir el optimizador

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma) # pasos del parametro de aprendizaje

    # train_list = []
    # test_list = []
    # list_accuracy = []
    # for epoch in range(1, args.epochs + 1):
    #     train_list.append(train(args, model, device, train_loader, optimizer, epoch))
    #     loss_test_list,accuracy_list = test(model, device, test_loader)
    #     test_list.append(loss_test_list)
    #     list_accuracy.append(accuracy_list)
    #     scheduler.step()

    # plot_loss_and_accuracy_in_one(train_list,list_accuracy, args.epochs)

    train_list = []
    test_list = []
    list_mae = []
    list_mse = []
    list_rmse = []
    #list_r2 = []

    for epoch in range(1, args.epochs + 1):
        train_list.append(train(args, model, device, train_loader, optimizer, epoch))
        loss_test_list, mae, mse, rmse = test(model, device, test_loader) # , r2
        test_list.append(loss_test_list)
        list_mae.append(mae)
        list_mse.append(mse)
        list_rmse.append(rmse)
        #list_r2.append(r2)
        scheduler.step()

    # Si tienes una función de visualización para estas métricas, puedes llamarla aquí
    plot_loss_and_metrics(train_list, test_list, list_mae, list_mse, list_rmse, args.epochs) # , list_r2

    testShow(model,device,test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "siamese_network_MSE_31_05_2024.pt")


if __name__ == "__main__":
    main()