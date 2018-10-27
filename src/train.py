import os
import sys
import time
import glob

from data import *
from model import *
from tensor_board_logger import TensorBoardLogger

SAVE_LOAD_CHECKPOINT = True
TEST_MODE = True


class Trainer(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_folder = "checkpoints"

    def __init__(self, model, data_loader, optimizer, criterion):
        self.model = model
        self.data_loader = data_loader.load()
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = TensorBoardLogger(os.path.join("logs", model.__class__.__name__))
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

    def evaluate(self):
        raise NotImplementedError

    @staticmethod
    def get_accuracy(target, output):
        _, argmax = torch.max(output, 1)
        accuracy = (target == argmax.squeeze()).float().mean()
        return accuracy


class RNNTrainer(Trainer):
    def __init__(self, model, data_loader, optimizer, criterion):
        super().__init__(model, data_loader, optimizer, criterion)

    def train(self, max_epoch, batch_size):
        print("Training started")
        self.model.train()
        epoch_resume = 0
        if SAVE_LOAD_CHECKPOINT:
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
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(device=self.device), target.to(device=self.device)
                self.optimizer.zero_grad()  # Initialize the gradient of model
                input_data = data.view(-1, batch_size, 1)   # (num of words / batch size) * batch size * index size(1)
                output, hidden, cell = self.model(input_data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                if TEST_MODE:
                    print("Train Epoch: {}/{} [{}/{} ({:.0f}%)]".format(
                        epoch, max_epoch, batch_idx * len(data), len(self.data_loader.dataset),
                                          100. * batch_idx / len(self.data_loader))
                    )
                    print("Loss: {:.6f}".format(loss.item()))
                    print("target : ", target)
                    print("output : ", output, end="\n\n")
                accuracy = self.get_accuracy(target, output)
                accuracy_sum += accuracy
                loss_sum += loss
            loss_avg = loss_sum / len(self.data_loader)
            accuracy_avg = accuracy_sum / len(self.data_loader)
            #TODO(kyungsoo): Make Tensorboard automatically execute when train.py runs if it is possible
            self.logger.log(loss_avg, accuracy_avg, self.model.named_parameters(), self.current_epoch)
            if SAVE_LOAD_CHECKPOINT:
                self.save_checkpoint({
                    "epoch": epoch + 1,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                })
        print("End")

    def evaluate(self, batch_size):
        print("Evaluation started")
        self.model.eval()
        epoch_resume = 0
        if SAVE_LOAD_CHECKPOINT:
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
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                input_data = data.view(-1, batch_size, 1)  # (num of words / batch size) * batch size * index size(1)
                output, _, _ = self.model(input_data)
                test_loss += self.criterion(output, target).item()  # sum up batch loss
                prediction = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += prediction.eq(target.view_as(prediction)).sum().item()
        test_loss /= len(self.data_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.data_loader.dataset),
            100. * correct / len(self.data_loader.dataset)))
        print("End")


def train_rnn_imdb(batch_size, learning_rate, max_epoch):
    acl_imdb = ACLIMDB(batch_size=batch_size, word_embedding='CBOW', is_eval=False, test_mode=TEST_MODE)
    # lstm = RnnImdb(torch.FloatTensor(acl_imdb.data.embedding_model.wv.vectors))
    lstm = RnnImdb(torch.from_numpy(acl_imdb.data.embedding_model.wv.vectors).type(torch.float))
    optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate, weight_decay=0.0003)
    criterion = nn.NLLLoss()
    trainer = RNNTrainer(lstm, acl_imdb, optimizer, criterion)
    trainer.train(max_epoch, batch_size)
    trainer.evaluate(batch_size)


def main():
    config = ConfigManager("RnnImdb").load()
    batch_size = int(config["BATCH_SIZE"])
    learning_rate = float(config["LEARNING_RATE"])
    max_epoch = int(config["MAX_EPOCH"])
    train_rnn_imdb(batch_size=batch_size, learning_rate=learning_rate, max_epoch=max_epoch)


if __name__ == "__main__":
    main()
