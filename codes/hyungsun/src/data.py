import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from datasets import Imdb


class BaseData(object):
    """ Base class of the data model.
    """

    def __init__(self):
        self.cuda = torch.cuda.is_available()


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
    data = None

    def __init__(self, batch_size, word_embedding, is_eval):
        BaseData.__init__(self)
        self.batch_size = batch_size
        self.word_embedding = word_embedding
        self.is_eval = is_eval
        self.data = Imdb(root=self.root, word_embedding=self.word_embedding, train=not self.is_eval)

    def load(self):
        additional_options = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        # TODO(hyungsun): make this class adapt word embedding dynamically.
        return torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True, **additional_options)


if __name__ == "__main__":
    # TODO(hyungsun): Remove these after debugging.
    loader = ACLIMDB(10, 'CBOW', False).load()
    for batch_idx, (data, target) in enumerate(loader):
        print(batch_idx)
