import os
import re
import torch
import numpy as np
from config import ConfigRNN
from gensim.models import KeyedVectors


def to_alphabetic(word):
    # All of br tags in our data is '<br />' with no exception.
    # So that word would be "blah<br" or "/>blah" after word2vec processing.
    # This cannot cover the case which looks like 'item.If'(No white space after period).
    # TODO(hyungsun): Find better way.
    cleaned_word = word.replace('<br', '').replace('/>', '')
    return re.sub('[^a-z ]+', '', cleaned_word.lower())


class Embed:
    embed_model_path = "embedmodel"
    embed_model_name = "embed_model.pickle"
    config = ConfigRNN.instance()
    embedding_model = None

    def __init__(self):
        # Load embedding model.
        try:
            saved_file = os.path.join(self.embed_model_path, self.embed_model_name)
            self.embedding_model = KeyedVectors.load(saved_file, mmap='r')
        except FileExistsError:
            pass
        words = self.embedding_model.wv.index2entity
        self.word_to_idx = {words[i]: i for i in range(0, len(words))}

    def get_embedding_model(self):
        return self.embedding_model

    def review2vec(self, review):
        word_vectors = []
        for word in review.split():
            print(word)
            alphabetic_word = to_alphabetic(word)
            if len(alphabetic_word) == 0:
                continue
            try:
                word_vectors.append([self.word_to_idx[alphabetic_word]])
            except KeyError:
                # print('An excluded word:', alphabetic_word)
                pass
        return torch.from_numpy(np.array([word_vectors])).long()


def main():
    sentence = "I Love movie"
    embed = Embed()
    print(embed.review2vec(sentence))


if __name__ == '__main__':
    main()
