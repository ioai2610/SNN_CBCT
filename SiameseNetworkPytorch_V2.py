import torch
import torch.nn as nn
import torchvision
import numpy as np
from utils import imshow
import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#################################
# DEFINICIÓN DE LA RED
#################################

class SiameseNetwork(nn.Module):
    
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(weights=None)

        # over-write the first conv layer to be able to read MNIST images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)
        
        return output
    

#################################
# PARA CARPETA DE TRAIN
#################################

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    # pérdida
    criterion = nn.MSELoss()

    # para graficar pérdida
    loss_train_list = []

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            loss_train_list.append(loss.item())
            if args.dry_run:
                break

    # Calcular la pérdida promedio para la epoch
    loss_train_list  = np.mean(loss_train_list)
    

    return loss_train_list

#################################
# PARA CARPETA DE TEST
#################################

# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0


#     criterion = nn.MSELoss()
#     # para graficar perdia en entrenamiento y accuracy
#     loss_test_list = []

#     with torch.no_grad():
#         for (images_1, images_2, targets) in test_loader:
#             images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
#             outputs = model(images_1, images_2).squeeze()
#             test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
 

#     test_loss /= len(test_loader.dataset)

#     loss_test_list.append(test_loss)

#     print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

#     loss_test_list = np.mean(loss_test_list)

#     return loss_test_list, []




def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    
    criterion = nn.MSELoss()
    loss_test_list = []

    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  
            
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    loss_test_list.append(test_loss)

    all_targets = np.array(all_targets)
    all_outputs = np.array(all_outputs)

    mae = mean_absolute_error(all_targets, all_outputs)
    mse = mean_squared_error(all_targets, all_outputs)
    rmse = np.sqrt(mse)
    #r2 = r2_score(all_targets, all_outputs)

    print(f'\nTest set: Average loss: {test_loss:.4f}')
    print(f'MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}\n') # , R^2: {r2:.4f}

    loss_test_list = np.mean(loss_test_list)

    return loss_test_list, mae, mse, rmse #, r2


#################################
# PARA MOSTRAR IMÁGENES
#################################

def testShow(model, device, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        for i, data in enumerate(test_loader,0): 
            images_1, images_2, targets = data
            #concat = torch.cat((images_1, images_2),0)
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss = criterion(outputs, targets) # sum up batch loss
            #plt.imshow(torchvision.utils.make_grid(concat)[0], cmap="grey")
            #plt.colorbar
            #plt.show()
            imshow(images_1,images_2,f"Métrica (0 - Diferente | 1 - Similares) : {test_loss.item()}")
            print(f"Métrica (0 - Diferente | 1 - Similares) : {test_loss.item()}")
            if i == 9:
                break