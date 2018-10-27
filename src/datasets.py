import os
import re
import pickle
import torch
import torch.utils.data as data
import numpy as np
from gensim.models import word2vec


TEST_DATA_SIZE = 16


def pad_sequence(sequences, batch_first=False, max_len=5000, padding_value=0):
    """Pad a list of variable length Tensors with zero
    See `torch.nn.utils.rnn.pad_sequence`
    """

    trailing_dims = sequences[0].size()[1:]
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def to_alphabetic(word):
    # All of br tags in our data is '<br />' with no exception.
    # So that word would be "blah<br" or "/>blah" after word2vec processing.
    # This cannot cover the case which looks like 'item.If'(No white space after period).
    # TODO(hyungsun): Find better way.
    cleaned_word = word.replace('<br', '').replace('/>', '')
    return re.sub('[^a-z ]+', '', cleaned_word.lower())


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
    # TODO(sejin): Make it load the value from the ini file
    embedding_dimension = 100

    # Pre-trained embedding model for nn.Embedding.
    # See https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding.from_pretrained
    embedding_model = None

    def __init__(self, root, word_embedding, train=True, debug=False):
        self.root = os.path.expanduser(root)
        self.word_embedding = word_embedding  # A string like 'CBOW', 'skip-gram'
        self.train = train  # training set or test set
        self.max_num_words = 0  # To make a 2-dimensional tensor with an uneven list of vectors
        self.test_mode = debug
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
            sentences=self.extract_sentences(),
            size=self.embedding_dimension,
            window=2,
            min_count=5,
            workers=12,
            iter=5,
            sg=sg,
        )

        if not self._check_processed() or self.test_mode:
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

    def extract_sentences(self):
        """Extract sentences from data set for Word2Vec model.
        See https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec for detail.

        :return: sentences type of list of list.
        """
        pickle_path = os.path.join(self.root, self.pickled_folder)
        pickle_file = 'sentences.pickle'

        if self.test_mode:
            try:
                os.remove(os.path.join(pickle_path, pickle_file))
                os.rmdir(pickle_path)
            except FileNotFoundError:
                pass

        try:
            with open(os.path.join(pickle_path, pickle_file), 'rb') as f:
                print("Sentences will be loaded from pickled file: " + pickle_file)
                return pickle.load(f)
        except FileNotFoundError:
            print("Cannot find pickled file to load sentences.")
            pass
        except Exception as error:
            raise error

        print("Extracting...")
        sentences = []
        for mode in ['train', 'test']:
            for classification in ['pos', 'neg', 'unsup']:
                if mode == 'test' and classification == 'unsup':
                    # There is no test/unsup in our data.
                    continue
                path = os.path.join(self.root, mode, classification)
                # sentences would be 12,500 review data sentences list.
                test_index = 0
                for sentence in word2vec.PathLineSentences(path):
                    test_index += 1
                    if self.test_mode and test_index > TEST_DATA_SIZE:
                        break

                    alphabetic_words = list(map(lambda x: to_alphabetic(x), sentence))
                    words = list(filter(lambda x: len(x) != 0, alphabetic_words))
                    sentences += words
        # Sentences look like [[review.split()], [...], ...].
        sentences = [sentences]
        try:
            os.mkdir(pickle_path)
        except FileExistsError:
            # 'processed' folder already exists.
            pass

        with open(os.path.join(pickle_path, pickle_file), 'wb') as f:
            pickle.dump(sentences, f, pickle.HIGHEST_PROTOCOL)

        print("Done.")
        return sentences

    def pre_process(self):
        """Select a pre-process function to execute and save the result in file system.
        """
        print("Processing...")
        words = self.embedding_model.wv.index2entity
        word_to_idx = {words[i]: i for i in range(0, len(words))}
        training_set, test_set = None, None
        for mode in ['train', 'test']:
            grades, vectors = [], []
            for classification in ['pos', 'neg']:
                for root, dirs, files in os.walk(os.path.join(self.root, mode, classification)):
                    test_index = 0
                    for file_name in files:
                        test_index += 1
                        if self.test_mode and test_index > TEST_DATA_SIZE:
                            break

                        # Get grade from filename such as "0_3.txt"
                        grade = 0 if int(file_name.split('_')[1][:-4]) > 5 else 1
                        grades.append(grade)

                        sentences = word2vec.LineSentence(os.path.join(root, file_name))
                        for sentence in sentences:
                            word_vectors = []
                            for word in sentence:
                                alphabetic_word = to_alphabetic(word)
                                if len(alphabetic_word) == 0:
                                    continue
                                try:
                                    word_vectors.append([word_to_idx[alphabetic_word]])
                                except KeyError:
                                    # print('An excluded word:', alphabetic_word)
                                    pass

                            self.max_num_words = max(self.max_num_words, len(word_vectors))
                            vectors.append(torch.from_numpy(np.array(word_vectors)).long())

            if mode == 'train':
                training_set = (pad_sequence(vectors, batch_first=True, max_len=self.max_num_words),
                                torch.from_numpy(np.array(grades)).long())
                pass
            else:
                training_set = (pad_sequence(vectors, batch_first=True, max_len=self.max_num_words),
                                torch.from_numpy(np.array(grades)).long())

        processed_folder_full_path = os.path.join(self.root, self.processed_folder)

        try:
            os.mkdir(processed_folder_full_path)
        except FileExistsError:
            # 'processed' folder already exists.
            pass

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        print("Done.")

    def pre_process_bow(self):
        """
        TODO(hyungsun): Implement with word2vec library.
        """
        raise NotImplementedError

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
