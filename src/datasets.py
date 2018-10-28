import os
import re
import pickle
import shutil

import torch
import shutil
import torch.utils.data as data
import numpy as np
from config import ConfigRNN
from gensim.models import word2vec
from nltk import word_tokenize

TEST_DATA_SIZE = 10


def pad_sequence(sequences, max_len, batch_first=False, padding_value=0):
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
    pickle_file = 'sentences.pickle'
    training_file = 'training.pt'
    test_file = 'test.pt'
    bow_file = 'labeledBow.feat'
    vocab_file = 'imdb.vocab'
    config = ConfigRNN.instance()

    # Constants for word embedding.
    # TODO(sejin): Make it load the value from the ini file
    embedding_dimension = 100

    # Pre-trained embedding model for nn.Embedding.
    # See https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding.from_pretrained
    embedding_model = None

    def __init__(self, root, embed_method, train=True, debug=False):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.max_num_words = 0  # To make a 2-dimensional tensor with an uneven list of vectors
        self.debug_mode = debug

        if self.debug_mode:
            self.processed_folder = 'debug_'+self.processed_folder
            self.pickled_folder = 'debug_'+self.pickled_folder

        self.pickle_path = os.path.join(self.root, self.pickled_folder)
        self.processed_path = os.path.join(self.root, self.processed_folder)

        if self.debug_mode:
            try:
                shutil.rmtree(self.pickle_path)
                shutil.rmtree(self.processed_path)
            except FileNotFoundError:
                pass

        if not self._check_exists():
            self.download()

        if embed_method == 'CBOW':
            sg = 0
        elif embed_method == 'SKIP_GRAM':
            sg = 1
        elif embed_method == 'DEFAULT':
            sg = None
        else:
            print(embed_method, "is not supported.")
            return

        if sg is None:
            words = self.extract_words()
            self.word_to_idx = {words[i]: i for i in range(0, len(words))}
        else:
            self.embedding_model = word2vec.Word2Vec(
                sentences=self.extract_sentences(),
                size=self.embedding_dimension,
                window=2,
                min_count=3,
                workers=12,
                iter=5,
                sg=sg,
            )
            words = self.embedding_model.wv.index2entity
            self.word_to_idx = {words[i]: i for i in range(0, len(words))}

        if not self._check_processed():
            self.pre_process(embed_method)

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

    def extract_words(self):
        pickle_path = os.path.join(self.root, self.pickled_folder)
        pickle_file = 'words.pickle'

        if self.debug_mode:
            try:
                shutil.rmtree(pickle_path)
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

        print("Extracting words...")
        words = set()
        for mode in ['train', 'test']:
            for classification in ['pos', 'neg', 'unsup']:
                if mode == 'test' and classification == 'unsup':
                    # There is no test/unsup in our data.
                    continue
                file_path = os.path.join(self.root, mode, classification)
                for root, dirs, files in os.walk(file_path):
                    test_index = 0
                    for file in files:
                        test_index += 1
                        if self.debug_mode and test_index > TEST_DATA_SIZE:
                            break
                        with open(os.path.join(file_path, file)) as f:
                            sentences = f.readlines()
                            for sentence in sentences:
                                new_words = set(word_tokenize(sentence))
                                words = words.union(new_words)
        alphabetic_words = []
        for word in words:
            word = to_alphabetic(word)
            if len(word) > 0:
                alphabetic_words.append(word)
        try:
            os.mkdir(pickle_path)
        except FileExistsError:
            # 'processed' folder already exists.
            pass

        with open(os.path.join(pickle_path, pickle_file), 'wb') as f:
            pickle.dump(words, f, pickle.HIGHEST_PROTOCOL)

        print("Done.")
        return alphabetic_words

    def extract_sentences(self):
        """Extract sentences from data set for Word2Vec model.
        See https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec for detail.

        :return: sentences type of list of list.
        """
        try:
            with open(os.path.join(self.pickle_path, self.pickle_file), 'rb') as f:
                print("Sentences will be loaded from pickled file: " + self.pickle_file)
                return pickle.load(f)
        except FileNotFoundError:
            print("Cannot find pickled file to load sentences.")
            pass
        except Exception as error:
            raise error

        print("Extracting sentences...")
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
                    if self.debug_mode and test_index > TEST_DATA_SIZE:
                        break

                    alphabetic_words = list(map(lambda x: to_alphabetic(x), sentence))
                    words = list(filter(lambda x: len(x) != 0, alphabetic_words))
                    sentences += words
        # Sentences look like [[review.split()], [...], ...].
        sentences = [sentences]
        try:
            os.mkdir(self.pickle_path)
        except FileExistsError:
            # 'processed' folder already exists.
            pass

        with open(os.path.join(self.pickle_path, self.pickle_file), 'wb') as f:
            pickle.dump(sentences, f, pickle.HIGHEST_PROTOCOL)

        print("Done.")
        return sentences

    def make_vectors_w2v(self, path):
        partial_vectors = []
        sentences = word2vec.LineSentence(path)
        for sentence in sentences:
            word_vectors = []
            for word in sentence:
                alphabetic_word = to_alphabetic(word)
                if len(alphabetic_word) == 0:
                    continue
                try:
                    word_vectors.append([self.word_to_idx[alphabetic_word]])
                except KeyError:
                    # print('An excluded word:', alphabetic_word)
                    pass
            self.max_num_words = max(self.max_num_words, len(word_vectors))
            partial_vectors.append(torch.from_numpy(np.array(word_vectors)).long())
        return partial_vectors

    def make_vectors_default(self, path):
        partial_vectors = []
        with open(path) as f:
            sentences = f.readlines()
            for sentence in sentences:
                word_vectors = []
                words = word_tokenize(sentence)
                for word in words:
                    alphabetic_word = to_alphabetic(word)
                    if len(alphabetic_word) == 0:
                        continue
                    try:
                        word_vectors.append([self.word_to_idx[alphabetic_word]])
                    except KeyError:
                        # print('An excluded word:', alphabetic_word)
                        pass

                self.max_num_words = max(self.max_num_words, len(word_vectors))
                partial_vectors.append(torch.from_numpy(np.array(word_vectors)).long())
        return partial_vectors

    def make_vectors_w2v(self, path):
        partial_vectors = []
        sentences = word2vec.LineSentence(path)
        for sentence in sentences:
            word_vectors = []
            for word in sentence:
                alphabetic_word = to_alphabetic(word)
                if len(alphabetic_word) == 0:
                    continue
                try:
                    word_vectors.append([self.word_to_idx[alphabetic_word]])
                except KeyError:
                    # print('An excluded word:', alphabetic_word)
                    pass
            self.max_num_words = max(self.max_num_words, len(word_vectors))
            partial_vectors.append(torch.from_numpy(np.array(word_vectors)).long())
        return partial_vectors

    def make_vectors_default(self, path):
        partial_vectors = []
        with open(path) as f:
            sentences = f.readlines()
            for sentence in sentences:
                word_vectors = []
                words = word_tokenize(sentence)
                for word in words:
                    alphabetic_word = to_alphabetic(word)
                    if len(alphabetic_word) == 0:
                        continue
                    try:
                        word_vectors.append([self.word_to_idx[alphabetic_word]])
                    except KeyError:
                        # print('An excluded word:', alphabetic_word)
                        pass
                self.max_num_words = max(self.max_num_words, len(word_vectors))
                partial_vectors.append(torch.from_numpy(np.array(word_vectors)).long())
        return partial_vectors

    def pre_process(self, embed_method):
        """Select a pre-process function to execute and save the result in file system.
        """
        print("Processing...")
        # Append pre-defined padding word to mask while training.
        # TODO(hyungsun): Masking word vectors.
        padding_value = 0
        training_set, test_set = None, None
        for mode in ['train', 'test']:
            grades, vectors = [], []
            for classification in ['pos', 'neg']:
                for root, dirs, files in os.walk(os.path.join(self.root, mode, classification)):
                    test_index = 0
                    for file_name in files:
                        test_index += 1
                        if self.debug_mode and test_index > TEST_DATA_SIZE:
                            break

                        # Get grade from filename such as "0_3.txt"
                        grade = 0 if int(file_name.split('_')[1][:-4]) > 5 else 1
                        grades.append(grade)
                        if embed_method == 'DEFAULT':
                            partial_vectors = self.make_vectors_default(os.path.join(root, file_name))
                            vectors.extend(partial_vectors)
                        else:
                            partial_vectors = self.make_vectors_w2v(os.path.join(root, file_name))
                            vectors.extend(partial_vectors)

            mode_set = (pad_sequence(vectors,
                                     max_len=self.max_num_words,
                                     batch_first=True,
                                     padding_value=padding_value),
                        torch.from_numpy(np.array(grades)).long())
            if mode == 'train':
                training_set = mode_set
            else:
                test_set = mode_set

        try:
            os.mkdir(os.path.join(self.root, self.processed_folder))
        except FileExistsError:
            # 'processed' folder already exists.
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
        fmt_str += '    Split: {}\n'.format('train' if self.train is True else 'test')
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str
