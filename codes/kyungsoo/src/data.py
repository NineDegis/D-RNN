import torch
from torchvision import datasets, transforms


class BaseData(object):
    """ Base class of the data model.

    All of data classes for network model should implement both eval_data and training_data.
    """

    def __init__(self):
        pass

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
    def __init__(self, batch_size):
        BaseData.__init__(self)
        self.batch_size = batch_size
        self.cuda = torch.cuda.is_available()

    def load(self, is_eval):
        additional_options = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        return torch.utils.data.DataLoader(
            datasets.MNIST(root='data/mnist',
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


class Reviews(BaseData):
    root = 'data/reviews/'
    train_file = 'train.txt'
    eval_file = 'eval.text'

    def __init__(self, batch_size):
        # TODO(Sejin): Implement.
        BaseData.__init__(self)
        self.batch_size = batch_size
        self.cuda = torch.cuda.is_available()



    def load(self, is_eval):
        pass

    def eval_data(self):
        pass

    def train_data(self):
        pass


