import torch.optim as optim
import time
from data import *
from model import *
import glob
import code
from config import *
from tensorBoardLogger import *


# TODO(hyungsun): Make this class more general.
class Trainer(object):
    def __init__(self, model, data_loader, optimizer, criterion, is_eval):
        self.logger = TensorBoardLogger('./logs/'+model.__class__.__name__)
        cuda = torch.cuda.is_available()
        self.model = model
        self.data_loader = data_loader.eval_data() if is_eval else data_loader.train_data()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.optimizer = optimizer
        self.prefix = "checkpoints/" + model.__class__.__name__ + "_"
        self.default_filename = self.prefix + str(int(time.time())) + ".pt"
        self.current_epoch = 0
        self.criterion = criterion

    # TODO(skrudtn): Make checkpoint for log
    def save_checkpoint(self):
        checkpoint = {
            "epoch": self.current_epoch + 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.default_filename)

    def load_checkpoint(self):
        file_names = glob.glob(self.prefix + "*.pt")
        if len(file_names) == 0:
            print("[!] Checkpoint not found.")
            return False
        file_name = file_names[-1]  # Pick the most recent file.
        checkpoint = torch.load(file_name)
        self.current_epoch = checkpoint["epoch"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.model.load_state_dict(checkpoint["model"])
        print("[+] Checkpoint Loaded. '{}'".format(file_name))
        return True

    def train(self, max_epoch):
        print("[+] Start training.")
        self.model.train()
        print("[+] Load Checkpoint if possible.")
        self.load_checkpoint()
        accuracy_sum = 0
        loss_sum = 0
        for max_epoch in range(1, max_epoch + 1):
            self.current_epoch = max_epoch
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                # variables = globals().copy()
                # variables.update(locals())
                # shell = code.InteractiveConsole(variables)
                # shell.interact()
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                accuracy = self.get_accuracy(target, output)
                accuracy_sum += accuracy
                loss_sum += loss

                if batch_idx % 100 == 0:
                    print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        max_epoch, batch_idx * len(data), len(self.data_loader.dataset),
                        100. * batch_idx / len(self.data_loader), loss.item()))
                    self.save_checkpoint()

            loss_avg = loss_sum/len(self.data_loader)
            accuracy_avg = accuracy_sum/len(self.data_loader)
            self.logger.log(loss_avg, accuracy_avg, self.model.named_parameters(), self.current_epoch)

        self.save_checkpoint()

    def evaluate(self):
        self.model.eval()
        self.load_checkpoint()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                prediction = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += prediction.eq(target.view_as(prediction)).sum().item()

        test_loss /= len(self.data_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.data_loader.dataset),
            100. * correct / len(self.data_loader.dataset)))

    @staticmethod
    def get_accuracy(target, output):
        _, argmax = torch.max(output, 1)
        accuracy = (target == argmax.squeeze()).float().mean()
        return accuracy


def train_cnn():
    model = Cnn()
    config = ConfigManager(model.__class__.__name__).load()
    optimizer = optim.SGD(model.parameters(), lr=float(config["LEARNING_RATE"]), momentum=float(config["MOMENTUM"]))
    criterion = torch.nn.NLLLoss()
    trainer = Trainer(model, MNIST(batch_size=10), optimizer, criterion, True)
    trainer.train(5)


def eval_cnn():
    model = Cnn()
    config = ConfigManager(model.__class__.__name__).load()
    optimizer = optim.SGD(model.parameters(), lr=float(config["LEARNING_RATE"]), momentum=float(config["MOMENTUM"]))
    criterion = torch.nn.NLLLoss()
    trainer = Trainer(model, MNIST(batch_size=10), optimizer, criterion, True)
    trainer.evaluate()


def train_rnn_imdb():
    # TODO(hyungsun): Verify this.
    config = ConfigManager(RnnImdb.__class__.__name__).load()
    model = RnnImdb()
    optimizer = optim.Adam(model.parameters(), lr=float(config["LEARNING_RATE"]))
    criterion = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model, ACLIMDB(batch_size=10), optimizer, criterion, True)
    trainer.train(5)


def main():
    pass

def open_debug_shell():
    """Open embedded interactive shell.

    Use this when you want to open shell to print or interact with variables in runtime.
    """
    variables = globals().copy()
    variables.update(locals())
    shell = code.InteractiveConsole(variables)
    shell.interact()


if __name__ == "__main__":
    main()
