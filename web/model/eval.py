import os
import sys
import time
import glob
import torch
import numpy as np
from model import RNN
from config import ConfigRNN
from embed import *

POSITIVE = "POS"
NEGATIVE = "NEG"


class Evaluator(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_folder = "checkpoints"

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.prefix = model.__class__.__name__ + "_"
        self.checkpoint_filename = self.prefix + str(int(time.time())) + ".pt"

    def load_checkpoint(self):
        root = os.path.dirname(sys.modules['__main__'].__file__)
        file_names = glob.glob(os.path.join(root, self.checkpoint_folder, self.prefix + "*.pt"))
        if len(file_names) == 0:
            print("[!] Checkpoint not found.")
            return {}
        file_name = file_names[-1]  # Pick the most recent file.
        print("[+] Checkpoint Loaded. '{}'".format(file_name))
        return torch.load(file_name)

    def evaluate(self, batch_size):
        raise NotImplementedError


class RNNEvaluator(Evaluator):
    config = ConfigRNN.instance()

    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)
        if self.config.LOGGING_ENABLE:
            from tensor_board_logger import TensorBoardLogger
            self.logger = TensorBoardLogger(os.path.join("logs", model.__class__.__name__))

        self.current_epoch = 0
        self.model.eval()

        # Load model & optimizer.
        checkpoint = self.load_checkpoint()
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.model.load_state_dict(checkpoint["model"])
        except KeyError:
            # There is no checkpoint
            pass

    def evaluate(self, review_vector):
        with torch.no_grad():
            review_vector= review_vector.to(self.device)

            input_data = review_vector.view(-1, 1, 1)  # (num of words / batch size) * batch size * index size(1)
            output, _, _ = self.model(input_data)
            prediction = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if prediction.eq(torch.tensor([1, 0])):
            return POSITIVE
        return NEGATIVE


def main():
    config = ConfigRNN.instance()
    # TODO(kyungsoo): Make this working.
    embedding_model = get_embedding_model()
    if config.EMBED_METHOD == "DEFAULT":
        vectors = embedding_model.wv.vectors

        # Add padding for masking.
        vectors = np.append(np.array([100 * [0]]), vectors, axis=0)
        model = RNN(torch.from_numpy(vectors).float())
    else:
        model = RNN()

    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    trainer = RNNEvaluator(model, optimizer)

    # TODO(kyungsoo): Make this working.
    review_vector = review2vec(sys.argv[0])
    print(trainer.evaluate(review_vector=review_vector))


if __name__ == "__main__":
    main()
