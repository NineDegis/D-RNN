import torch
from data_loader import load_mnist_data
from neural_network import NeuralNetwork
import tests

class InitDataSet:
    def __init__(self, batch_size=64, test_batch_size=1000, epochs=10, lr=0.01, momentum=0.5, no_cuda=True, seed=1,
                 log_interval=10, ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.no_cuda = no_cuda
        self.seed = seed
        self.log_interval = log_interval

if __name__ == '__main__':
    init_data = InitDataSet()

    train_loader, test_loader = load_mnist_data(init_data)
    tests.load_test(train_loader)
    tests.load_test(test_loader)

    nn = NeuralNetwork(init_data)
    nn.train(train_loader)
    # tests.nn_test(nn)
    # nn.train()
