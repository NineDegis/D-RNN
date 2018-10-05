import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from datasets import Imdb


class BaseData(object):
    """ Base class of the data model.

    All of data classes for network model should implement both eval_data and training_data.
    """

    def __init__(self):
        self.cuda = torch.cuda.is_available()

    def eval_data(self):
        """ Return data for network model to evaluate.

        :return: Data sets for evaluation.
        """
        raise NotImplementedError("[-] Function 'eval_data' not implemented at " + self.__class__.__name__)

    def train_data(self):
        """ Return data for network model to train.

        :return: Data sets for training.
        """
        raise NotImplementedError("[-] Function 'train_data' not implemented at " + self.__class__.__name__)


class MNIST(BaseData):
    root = 'data/mnist'

    def __init__(self, batch_size):
        BaseData.__init__(self)
        self.batch_size = batch_size

    def load(self, is_eval):
        additional_options = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        return torch.utils.data.DataLoader(
            datasets.MNIST(root=self.root,
                           train=not is_eval,
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.batch_size,
            shuffle=True,
            **additional_options)

    def eval_data(self):
        return self.load(True)

    def train_data(self):
        return self.load(False)


class ACLIMDB(BaseData):
    root = 'data/aclImdb/'

    def __init__(self, batch_size):
        BaseData.__init__(self)
        self.batch_size = batch_size

    def load(self, is_eval):
        additional_options = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        return torch.utils.data.DataLoader(
            Imdb(root=self.root, word_embedding='CBOW', train=not is_eval),
            batch_size=self.batch_size,
            shuffle=True,
            **additional_options)

    def eval_data(self):
        return self.load(True)

    def train_data(self):
        return self.load(False)


if __name__ == "__main__":
    # TODO(hyungsun): Remove these after debugging.
    loader = ACLIMDB(10).load(False)
    for batch_idx, (data, target) in enumerate(loader):
        print(batch_idx, data.shape)
