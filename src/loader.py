import os
import torch
import torch.utils.data as data
from datasets import Imdb
from config import ConfigRNN


class BaseData(object):
    """ Base class of the data model.
    """
    def __init__(self):
        self.cuda = torch.cuda.is_available()


class ACLIMDB(BaseData):
    # root = 'data/aclImdb/'
    config = ConfigRNN.instance()
    root = os.path.join('data', 'aclImdb')
    data = None

    def __init__(self, batch_size, embed_method, is_eval, debug):
        BaseData.__init__(self)
        self.batch_size = batch_size
        self.embed_method = embed_method
        self.data = Imdb(
            root=self.root,
            embed_method=self.embed_method,
            train=not is_eval,
            debug=debug)

    def load(self):
        additional_options = {'num_workers': 0, 'pin_memory': True} if self.cuda else {}
        # TODO(hyungsun): make this class adapt word embedding dynamically.
        return torch.utils.data.DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=self.config.SHUFFLE,
            drop_last=True,
            **additional_options)
