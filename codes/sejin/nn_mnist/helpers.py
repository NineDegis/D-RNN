import numpy as np
import torch

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)   # Prevent overflow
    return exp_x / np.sum(exp_x)

def onehot(x):
    if x >= 10:
        return None
    onehot_vector = torch.zeros(1, 10)
    onehot_vector[0][x] = 1
    return onehot_vector