import torch
from torchvision import datasets, transforms


class MNIST(object):
    @staticmethod
    def load(is_eval, batch_size, use_cuda):
        additional_options = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            datasets.MNIST(root='data/mnist',
                           train=not is_eval,
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size,
            shuffle=True,
            **additional_options)
