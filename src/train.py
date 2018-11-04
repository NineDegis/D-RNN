import os
import sys
import time
import glob
import torch
import numpy as np
from model import RNN
from loader import ACLIMDB
from config import ConfigRNN


class Trainer(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_folder = "checkpoints"

    def __init__(self, model, data_loader, optimizer):
        self.data_loader = data_loader.load()
        self.model = model
        self.optimizer = optimizer
        self.prefix = model.__class__.__name__ + "_"
        self.checkpoint_filename = self.prefix + str(int(time.time())) + ".pt"
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    # TODO(kyungsoo): Make checkpoint for log
    def save_checkpoint(self, checkpoint):
        try:
            os.mkdir(self.checkpoint_folder)
        except FileExistsError:
            pass
        torch.save(checkpoint, os.path.join(self.checkpoint_folder, self.checkpoint_filename))

    def load_checkpoint(self):
        root = os.path.dirname(sys.modules['__main__'].__file__)
        file_names = glob.glob(os.path.join(root, self.checkpoint_folder, self.prefix + "*.pt"))
        if len(file_names) == 0:
            print("[!] Checkpoint not found.")
            return {}
        file_name = file_names[-1]  # Pick the most recent file.
        print("[+] Checkpoint Loaded. '{}'".format(file_name))
        return torch.load(file_name)

    def train(self, max_epoch):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    @staticmethod
    def get_accuracy(target, output):
        _, argmax = torch.max(output, 1)
        accuracy = (target == argmax.squeeze()).float().mean()
        return accuracy


class RNNTrainer(Trainer):
    config = ConfigRNN.instance()

    def __init__(self, model, data_loader, optimizer):
        super().__init__(model, data_loader, optimizer)
        if self.config.BOARD_LOGGING:
            from tensor_board_logger import TensorBoardLogger
            self.logger = TensorBoardLogger(os.path.join("logs", model.__class__.__name__))

        self.current_epoch = 0

    def train(self, max_epoch):
        print("Training started")

        # Set model to train mode.
        self.model.train()
        epoch_resume = 0
        if self.config.SAVE_CHECKPOINT:
            checkpoint = self.load_checkpoint()
            try:
                epoch_resume = checkpoint["epoch"]
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.model.load_state_dict(checkpoint["model"])
            except KeyError:
                # There is no checkpoint
                pass
        for epoch in range(epoch_resume, max_epoch):
            accuracy_sum = 0
            loss_sum = 0
            self.current_epoch = epoch
            for batch_idx, (_data, target) in enumerate(self.data_loader):
                # Transpose vector to make it (num of words / batch size) * batch size * index size(1).
                _data = np.transpose(_data, (1, 0, 2))
                _data, target = _data.to(device=self.device), target.to(device=self.device)

                # Initialize the gradient of model
                self.optimizer.zero_grad()
                output, hidden, cell, sorted_target = self.model(_data, target)
                loss = self.config.CRITERION(output, sorted_target)
                loss.backward()
                self.optimizer.step()
                if self.config.DEBUG_MODE:
                    print("Train Epoch: {}/{} [{}/{} ({:.0f}%)]".format(
                        epoch, max_epoch, batch_idx * _data.shape[1],
                        len(self.data_loader.dataset), 100. * batch_idx / len(self.data_loader)))
                    print("Loss: {:.6f}".format(loss.item()))
                    print("target : ", target)
                    print("output : ", output, end="\n\n")
                accuracy = self.get_accuracy(sorted_target, output)
                accuracy_sum += accuracy
                loss_sum += loss
            if self.config.BOARD_LOGGING:
                if len(self.data_loader) == 0:
                    raise Exception("Data size is smaller than batch size.")
                loss_avg = loss_sum / len(self.data_loader)
                accuracy_avg = accuracy_sum / len(self.data_loader)
                # TODO(kyungsoo): Make Tensorboard automatically execute when train.py runs if it is possible
                self.logger.log(loss_avg, accuracy_avg, self.model.named_parameters(), self.current_epoch)
                self.save_checkpoint({
                    "epoch": epoch + 1,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                })
        print("End")

    def evaluate(self):
        print("Evaluation started")

        # Set model to eval mode.
        self.model.eval()
        if self.config.SAVE_CHECKPOINT:
            checkpoint = self.load_checkpoint()
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.model.load_state_dict(checkpoint["model"])
            except KeyError:
                # There is no checkpoint
                pass

        correct = 0
        with torch.no_grad():
            for _data, target in self.data_loader:
                # Transpose vector to make it (num of words / batch size) * batch size * index size(1).
                _data = np.transpose(_data, (1, 0, 2))
                _data, target = _data.to(device=self.device), target.to(device=self.device)

                # Initialize the gradient of model
                self.optimizer.zero_grad()
                output, hidden, cell, sorted_target = self.model(_data, target)

                _, argmax = torch.max(output, 1)
                correct += (sorted_target == argmax.squeeze()).nonzero().size(0) / 2

        print('\nAccuracy: {:.0f}%\n'.format(correct / len(self.data_loader.dataset)))
        print("End")


def main():
    config = ConfigRNN.instance()
    loader = ACLIMDB(
        batch_size=config.BATCH_SIZE,
        embed_method=config.EMBED_METHOD,
        is_eval=config.EVAL_MODE,
        debug=config.CONSOLE_LOGGING)
    embedding_model = loader.data.embedding_model
    if embedding_model == "DEFAULT":
        model = RNN()
    else:
        vectors = loader.data.embedding_model.wv.vectors

        # Add padding for masking.
        vectors = np.append(np.array([100 * [0]]), vectors, axis=0)
        model = RNN(torch.from_numpy(vectors).float())

    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    trainer = RNNTrainer(model, loader, optimizer)
    if config.EVAL_MODE:
        trainer.evaluate()
    else:
        trainer.train(config.MAX_EPOCH)


if __name__ == "__main__":
    main()
