import argparse
import torch
from torchvision import datasets, transforms

def init_arguments(init_data):
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=init_data.batch_size, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=init_data.test_batch_size, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=init_data.epochs, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=init_data.lr, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=init_data.momentum, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=init_data.no_cuda,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=init_data.seed, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=init_data.log_interval, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    kwargs = {'num_workers': 1, 'pin_memory': True}
    return args, kwargs

def load_mnist_data(init_data):
    args, kwargs = init_arguments(init_data)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader