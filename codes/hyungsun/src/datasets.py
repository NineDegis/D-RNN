import os

import torch
import torch.utils.data as data
from gensim.models import word2vec
import nltk
from nltk import word_tokenize


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

    def __init__(self, root, word_embedding, train=True):
        self.root = os.path.expanduser(root)
        self.word_embedding = word_embedding  # A string like 'CBOW', 'skip-gram'
        self.train = train  # training set or test set
        self.max_num_words = 0  # To make a 2-dimensional tensor with an uneven list of vectors

        self.embedding_dimension = 100
        self.window_size = 2
        self.min_count = 5
        self.num_workers = 12
        self.iter = 5  # 100

        if not self._check_exists():
            self.download()

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

    def rectangularize(self, nested_list, fill=0):
        """Make a given nested list in the form of a rectangle"""
        max_len = len(max(nested_list, key=len))
        for flat in nested_list:
            extender = [fill] * (max_len - len(flat))
            flat.extend(extender)
        return nested_list

    def extract_words(self):
        """

        :return: a dictionary of lists of tokenized words, it will look like this.
            {
                'train': [...],
                'test': [...],
            }
        """
        extracted_words = {}
        for mode in ['train', 'test']:
            sentences_in_mode = []
            for classification in ['pos', 'neg', 'unsup']:
                path = os.path.join(self.root, mode, classification)
                try:
                    movie_eval_list = list(word2vec.PathLineSentences(path))
                except ValueError:
                    # There is no test/unsup path.
                    pass
                sentences_in_mode = movie_eval_list
            extracted_words[mode] = sentences_in_mode
        return extracted_words

    def pre_process(self, word_embedding):
        """Select a pre-process function to execute and save the result in file system.

        :param word_embedding: A string that nominates which method of word embedding algorithms would be used.
        """
        supportable_word_embeddings = {
            'name': ['BOW', 'CBOW', 'skip-gram'],
            'value': [None, 0, 1],
        }
        if word_embedding not in supportable_word_embeddings['name']:
            print(word_embedding, "is not supported.")
            print("You can use", supportable_word_embeddings['name'], "as a word embedding algorithm.")
            return

        print("Processing...")
        if word_embedding == supportable_word_embeddings['name'][0]:
            training_set, test_set = self.pre_process_sparse()
        else:
            word_embedding_name_idx = supportable_word_embeddings['name'].index(word_embedding)
            word_embedding_no = supportable_word_embeddings['value'][word_embedding_name_idx]
            training_set, test_set = self.pre_process_dense(word_embedding_no)

        self.save_processing_cache(training_set, test_set)
        print("Done.")

    def pre_process_sparse(self):
        """
        Pre-processing with BOW.

        TODO(hyungsun): Try various embedding techniques. Candidate: CBOW, N-GRAM
        """
        training_set, test_set = None, None
        with open(os.path.join(self.root, self.vocab_file), 'r') as f:
            dictionary = {vocab: i for i, vocab in enumerate(f.readlines())}
            size = len(dictionary)
        for mode in ['train', 'test']:
            grades, vectors = [], []
            with open(os.path.join(self.root, mode, self.bow_file), 'r') as f:
                for line in f.readlines():
                    sparse_bow = line.split()
                    grades.append(int(sparse_bow[0]))
                    vector = [0] * size
                    for token in sparse_bow[1:]:
                        token = token.split(":")
                        vector[int(token[0])] = int(token[1])
                    vectors.append(vector)
            if mode == 'train':
                training_set = (torch.tensor(vectors, dtype=torch.long), torch.tensor(grades))
            else:
                test_set = (torch.tensor(vectors, dtype=torch.long), torch.tensor(grades))
        return training_set, test_set

    def pre_process_dense(self, word_embedding_no):
        """Pre-processing with dense-represented bow which is produced by word2vec

        :param word_embedding_no: 0 for CBOW, 1 for skip-gram.
        :return: A 2-length tuple. It is consisted of word vectors and grades.
        """
        extracted_words = self.extract_words()
        word_vector = word2vec.Word2Vec(
            # word2vec_bow['train']['vectors'] + word2vec_bow['test']['vectors'],
            extracted_words['train'] + extracted_words['test'],
            size=self.embedding_dimension,
            window=self.window_size,
            min_count=self.min_count,
            workers=self.num_workers,
            iter=self.iter,
            sg=word_embedding_no,
        )
        dense_bow = word_vector.wv  # vectors of words
        del word_vector

        print(len(dense_bow.index2entity))
        training_set, test_set = None, None
        for mode in ['train', 'test']:
            grades, vectors = [], []
            for classification in ['pos', 'neg']:
                path = os.path.join(self.root, mode, classification)
                for root, dirs, files in os.walk(path):
                    for idx, file_name in enumerate(files):
                        grades.append(file_name.rsplit('.', 1)[0].rsplit('_', 1)[1])
                        full_file_path = os.path.join(root, file_name)
                        words = word2vec.LineSentence(full_file_path)
                        word_vectors = []
                        for idx, word in enumerate(list(words)[0]):
                            try:
                                word_vectors.append(dense_bow.get_vector(word))
                            except KeyError:
                                # print('An excluded word:', word)
                                pass
                        vectors.append(word_vectors)
            vectors = self.rectangularize(vectors, 0)
            print(len(vectors))
            print(len(vectors[0]), len(vectors[1]), len(vectors[2]))
            print(len(grades))
            if mode == 'train':
                training_set = torch.tensor(vectors, dtype=torch.long), torch.tensor(grades)
            else:
                test_set = torch.tensor(vectors, dtype=torch.long), torch.tensor(grades)
        return training_set, test_set

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
