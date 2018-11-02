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

    def train(self, max_epoch, batch_size):
        raise NotImplementedError

    def evaluate(self, batch_size):
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
        if self.config.LOGGING_ENABLE:
            from tensor_board_logger import TensorBoardLogger
            self.logger = TensorBoardLogger(os.path.join("logs", model.__class__.__name__))

        self.current_epoch = 0

    def train(self, max_epoch, batch_size):
        print("Training started")
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # Set model to train mode.
        self.model.train()
        epoch_resume = 0
        if self.config.CHECKPOINT_ENABLE:
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
            if self.config.LOGGING_ENABLE:
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

    def evaluate(self, batch_size):
        print("Evaluation started")

        # Set model to eval mode.
        self.model.eval()
        if self.config.CHECKPOINT_ENABLE:
            checkpoint = self.load_checkpoint()
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.model.load_state_dict(checkpoint["model"])
            except KeyError:
                # There is no checkpoint
                pass

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for _data, target in self.data_loader:
                _data, target = _data.to(self.device), target.to(self.device)
                input_data = _data.view(-1, batch_size, 1)  # (num of words / batch size) * batch size * index size(1)
                output, _, _ = self.model(input_data)
                test_loss += self.config.CRITERION(output, target).item()  # sum up batch loss
                prediction = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += prediction.eq(target.view_as(prediction)).sum().item()
        test_loss /= len(self.data_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.data_loader.dataset),
            100. * correct / len(self.data_loader.dataset)))
        print("End")


def main():
    config = ConfigRNN.instance()
    loader = ACLIMDB(
        batch_size=config.BATCH_SIZE,
        embed_method=config.EMBED_METHOD,
        is_eval=False,
        debug=config.DEBUG_MODE)
    embedding_model = loader.data.embedding_model
    if embedding_model == "DEFAULT":
        vectors = loader.data.embedding_model.wv.vectors

        # Add padding for masking.
        vectors = np.append(np.array([100 *  [0]]), vectors, axis=0)
        model = RNN(torch.from_numpy(vectors).float())
    else:
        model = RNN()

    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    trainer = RNNTrainer(model, loader, optimizer)
    trainer.train(config.MAX_EPOCH, config.BATCH_SIZE)


if __name__ == "__main__":
    main()
