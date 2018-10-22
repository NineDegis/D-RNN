import os

from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import nltk
from nltk import word_tokenize
from gensim.test.utils import datapath

nltk.download('punkt')


def read_data_files(path):
    """Read strings from data files.

    :param path: A string, ex) ./data/train/pos
    :return: A list of strings, each element contains all strings from a single data file.
    """
    input_data = []
    for root, dirs, files in os.walk(path):
        for idx, file_name in enumerate(files):
            if idx == 1000:
                break
            full_file_path = os.path.join(root, file_name)
            with open(full_file_path) as file:
                try:
                    line = file.readlines()
                    input_data += line
                except UnicodeDecodeError:
                    print("UnicodeDecodeError has occurred. Ignore the strange word...")
    return input_data


def make_word2vec_model(input_data):
    """Convert a list of strings into a word2vec model.

    :param input_data: a list of strings
    :return: a word2vec model
    """

    # Tokenize read strings into words
    # words = []
    # for each_str in input_data:
    #     words.append(word_tokenize(each_str))
    # print('-' * 10)
    # print('length of words:', len(words))
    # print('length of words[0]:', len(words[0]))
    # print('words[0]:', words[0])

    # Convert words into vectors
    model = Word2Vec(
        # words,
        input_data,
        # size=100,
        # window = 2,
        # min_count=50,
        # workers=4,
        # iter=100,
        sg=1    # skip-gram
    )

    return model


if __name__ == '__main__':
    saved_model_name = 'model.wv'

    # Load a model
    try:
        model = KeyedVectors.load(saved_model_name, mmap='r')
    except(FileNotFoundError):
        # input_data = read_data_files('./data/aclImdb/train/pos')
        # input_data = word2vec.Text8Corpus(datapath('./data/aclImdb/train/pos'))
        sentences = word2vec.PathLineSentences(
            # datapath(
            #     os.path.expanduser(
            #         os.path.join('data', 'aclImdb', 'train', 'pos')
            #     )
            # )
            'C:\\Users\\jinai\\git_projects\\D-RNN\\experimental\\sejin\\word2vec_test\\data\\aclImdb\\test\\pos'
        )
        model = make_word2vec_model(list(sentences))
        model.save(saved_model_name)

    # Remove unnecessary data from the memory
    model.init_sims(replace=True)

    wv = model.wv
    wv_list = wv.index2entity
    print(wv_list)
    print(len(wv_list))
    print(wv.get_vector('scary'))
    print(len(wv.get_vector('scary')))
    print(wv.get_vector('what would you do if there is no word that I am finding'))

    # Get some meaningful results
    print(model.wv.similarity('he', 'she'))
    print(model.most_similar(positive=['scary'], topn=10))