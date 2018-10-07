import torch
from helpers import sigmoid, onehot


class NeuralNetwork:
    def __init__(self, init_data):
        self.init_data = init_data

    def hypothesis(self, x, w, b=0):
        return sigmoid(torch.mm(x, w) + b)

    # def dLoss(self, w, x, y):
    #     if y == 1:
    #         return -math.log(self.hypothesis(w, x), math.e)
    #     elif y == 0:
    #         return -math.log(1 - self.hypothesis(w, x), math.e)

    def train(self, train_loader):
        layer_sizes = [28*28, 100, 50, 10]
        torch.manual_seed(self.init_data.seed)
        weights = []
        for i in range(1, len(layer_sizes)):
            weights.append(torch.randn(layer_sizes[i - 1], layer_sizes[i]))

        # for epoch in range(self.init_data.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > 1:
                break

            inputs = data.view(64, 1, -1)
            for idx, input in enumerate(inputs):
                # forward propagation
                hypo_list = [input]
                for idx, weight in enumerate(weights):
                    hypo_list.append(self.hypothesis(hypo_list[idx], weight))

                # back propagation
                d3 = hypo_list[3] - onehot(target[idx])
                weights[2] -= self.init_data.lr * torch.mm(hypo_list[2].transpose(0, 1), d3)    # weight = weight - lr * dcost
                d2 = d3 * weights[1] * hypo_list[2] * (1 - hypo_list[2])
                weights[1] -= self.init_data.lr * torch.mm(hypo_list[1].transpose(0, 1), d2)
                d1 = d2 * weights[0] * hypo_list[1] * (1 - hypo_list[1])
                weights[0] -= self.init_data.lr * torch.mm(hypo_list[0].transpose(0, 1), d1)

                d = []
                for i in