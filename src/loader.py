import os
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from datasets import Imdb


class BaseData(object):
    """ Base class of the data model.
    """
    def __init__(self):
        self.cuda = torch.cuda.is_available()


class MNIST(BaseData):
    """ For now, we do not use this class.
    """
    root = 'data/mnist'

    def __init__(self, batch_size):
        BaseData.__init__(self)
        self.batch_size = batch_size

    def load(self, is_eval):
        additional_options = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        dataset = datasets.MNIST(root=self.root,
                                 train=not is_eval,
                                 download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **additional_options)

    def eval_data(self):
        return self.load(True)

    def train_data(self):
        return self.load(False)


class ACLIMDB(BaseData):
    # root = 'data/aclImdb/'
    root = os.path.join('data', 'aclImdb')
    data = None

    def __init__(self, batch_size, word_embedding, is_eval, debug):
        BaseData.__init__(self)
        self.batch_size = batch_size
        self.word_embedding = word_embedding
        self.data = Imdb(
            root=self.root,
            word_embedding=self.word_embedding,
            train=not is_eval,
            debug=debug)

    def load(self):
        additional_options = {'num_workers': 0, 'pin_memory': True} if self.cuda else {}
        # TODO(hyungsun): make this class adapt word embedding dynamically.
        return torch.utils.data.DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            **additional_options)


if __name__ == '__main__':
    ACLIMDB(5, 'CBOW', False, True).load()
