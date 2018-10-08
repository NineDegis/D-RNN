import os
import re
import pickle

import torch
import torch.utils.data as data
from gensim.models import word2vec


class Imdb(data.Dataset):
    """Note: This class assume that the data is in root directory already just for now.
     Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
    """
    processed_folder = 'processed'
    pickled_folder = 'pickled'
    training_file = 'training.pt'
    test_file = 'test.pt'
    bow_file = 'labeledBow.feat'
    vocab_file = 'imdb.vocab'

    # Constants for word embedding.
    embedding_dimension = 100

    # Pre-trained embedding model for nn.Embedding.
    # See https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding.from_pretrained
    embedding_model = None

    def __init__(self, root, word_embedding, train=True):
        self.root = os.path.expanduser(root)
        self.word_embedding = word_embedding  # A string like 'CBOW', 'skip-gram'
        self.train = train  # training set or test set
        self.max_num_words = 0  # To make a 2-dimensional tensor with an uneven list of vectors
        if not self._check_exists():
            self.download()

        if word_embedding == 'CBOW':
            sg = 0
        elif word_embedding == 'SKIP_GRAM':
            sg = 1
        else:
            print(word_embedding, "is not supported.")
            return

        self.embedding_model = word2vec.Word2Vec(
            self.extract_words(),
            size=self.embedding_dimension,
            window=2,
            min_count=5,
            workers=12,
            iter=5,
            sg=sg,
        )

        if not self._check_processed():
            self.pre_process(word_embedding)
        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
            print(len(self.train_data), len(self.train_labels))

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

    def extract_words(self):
        pickle_path = os.path.join(self.root, self.pickled_folder)
        pickle_file = 'words.pickle'
        try:
            with open(os.path.join(pickle_path, pickle_file), 'rb') as f:
                return list(pickle.load(f))
        except FileNotFoundError:
            print("Pickled words file not found.")
            pass
        except Exception as error:
            raise error

        print("Started to extract words from data.")
        word_set = set()
        for mode in ['train', 'test']:
            for classification in ['pos', 'neg', 'unsup']:
                if mode == 'test' and classification == 'unsup':
                    # There is no test/unsup in our data.
                    continue
                path = os.path.join(self.root, mode, classification)
                for file in os.listdir(path):
                    with open(os.path.join(path, file), 'r') as f:
                        for sentence in f.readlines():
                            for word in sentence.split():
                                alphabetic_word = re.sub('[^A-Za-z]+', '', word).lower()
                                if word == 'br' or alphabetic_word == '':
                                    continue
                                word_set.add(alphabetic_word)
        result = list(word_set)

        try:
            os.mkdir(pickle_path)
        except FileExistsError:
            # 'processed' folder already exists.
            pass

        with open(os.path.join(pickle_path, pickle_file), 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

        print("Extraction done.")
        return result

    def pre_process(self):
        """Select a pre-process function to execute and save the result in file system.

        :param word_embedding: A string that nominates which method of word embedding algorithms would be used.
        """
        print("Processing...")
        training_set, test_set = None, None
        for mode in ['train', 'test']:
            grades, vectors = [], []
            for classification in ['pos', 'neg']:
                for root, dirs, files in os.walk(os.path.join(self.root, mode, classification)):
                    for idx, file_name in enumerate(files):
                        grades.append(file_name.rsplit('.', 1)[0].rsplit('_', 1)[1])
                        words = word2vec.LineSentence(os.path.join(root, file_name))
                        word_vectors = []
                        for _, word in enumerate(list(words)[0]):
                            alphabetic_word = re.sub('[^A-Za-z]+', '', word).lower()
                            if alphabetic_word == '' or alphabetic_word == 'br':
                                continue
                            try:
                                word_vectors.append(self.embedding_model.wv.get_vector(alphabetic_word).tolist())
                            except KeyError:
                                # print('An excluded word:', alphabetic_word)
                                pass
                        vectors.append(word_vectors)
            if mode == 'train':
                training_set = vectors, grades
            else:
                test_set = vectors, grades

        self.save_processing_cache(training_set, test_set)
        print("Done.")

    def pre_process_bow(self):
        """
        TODO(hyungsun): Implement with word2vec library.
        """
        raise NotImplementedError

    def save_processing_cache(self, training_set, test_set):
        processed_folder_full_path = os.path.join(self.root, self.processed_folder)

        # Create a folder for processing cache files.
        try:
            os.mkdir(processed_folder_full_path)
        except FileExistsError:
            # 'processed' folder already exists.
            pass

        # Save
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

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

    def _check_embed_model_created(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.embedding_file))

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
