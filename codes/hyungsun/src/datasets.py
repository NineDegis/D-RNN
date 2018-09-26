import torch
import torch.utils.data as data
import os


class Imdb(data.Dataset):
    """Note: This class assume that the data is in root directory already just for now.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
    """
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    bow_file = 'labeledBow.feat'
    vocab_file = 'imdb.vocab'

    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        if not self._check_exists():
            self.download()

        if not self._check_processed():
            self.pre_process()

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (vector, target) where target is index of the target class.
        """
        if self.train:
            vector, target = self.train_data[index], self.train_labels[index]
        else:
            vector, target = self.test_data[index], self.test_labels[index]
        return vector, target

    def pre_process(self):
        """
        Pre-processing with BOW.

        TODO(hyungsun): Try various embedding techniques. Candidate: CBOW, N-GRAM
        """
        print("Processing...")
        with open(os.path.join(self.root, self.vocab_file), 'r') as f:
            dictionary = {vocab: i for i, vocab in enumerate(f.readlines())}

        for mode in ['train', 'test']:
            grades, vectors = [], []
            with open(os.path.join(self.root, mode, self.bow_file), 'r') as f:
                size = len(dictionary)
                for line in f.readlines():
                    bow = line.split()
                    grades.append(int(bow[0]))
                    vector = [0] * size
                    for token in bow[1:]:
                        token = token.split(":")
                        vector[int(token[0])] = int(token[1])
                    vectors.append(vector)
            if mode == 'train':
                training_set = (torch.tensor(vectors, dtype=torch.long), torch.Tensor(grades))
            else:
                test_set = (torch.tensor(vectors, dtype=torch.long), torch.Tensor(grades))
        try:
            os.mkdir(os.path.join(self.root, self.processed_folder))
        except FileExistsError:
            pass
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        print("Done.")

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        """Check if the dataset is downloaded."""
        return os.path.exists(os.path.join(self.root))

    def _check_processed(self):
        """Check if the dataset is preprocessed."""
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the Imdb review data if it doesn't exist in processed_folder already."""
        # TODO(hyungsun): Implement if needed.
        pass

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str
