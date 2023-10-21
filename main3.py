import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from utils import plot_loss_and_accuracy_in_one
from torch.optim.lr_scheduler import StepLR
from SiameseNetworkPytorch import SiameseNetwork, train, test, testShow
from DicomDataset import DicomDataset
import numpy as np

train_csv_path = "./training_data.csv"
test_csv_path = "./test_data.csv"
root = "./"


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dataset = DicomDataset(root, train_csv_path, transforms.Compose([transforms.Resize(105,antialias=True)]))
    test_dataset = DicomDataset(root, test_csv_path, transforms.Compose([transforms.Resize(255, antialias=True)]))
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    train_list = []
    test_list = []
    list_accuracy = []
    for epoch in range(1, args.epochs + 1):
        train_list.append(train(args, model, device, train_loader, optimizer, epoch))
        loss_test_list,accuracy_list = test(model, device, test_loader)
        test_list.append(loss_test_list)
        list_accuracy.append(accuracy_list)
        scheduler.step()

    plot_loss_and_accuracy_in_one(train_list,list_accuracy, args.epochs)
       
    testShow(model,device,test_loader)
    if args.save_model:
        torch.save(model.state_dict(), "siamese_network_BCE.pt")


if __name__ == "__main__":
    main()