import sys
import time
import glob
import torch
import os
from model import ReviewParser
from config import ConfigRNN
from embed import Embed


class Evaluator(object):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_folder = "checkpoints"

    def __init__(self, model):
        self.model = model
        self.prefix = model.__class__.__name__ + "_"
        self.checkpoint_filename = self.prefix + str(int(time.time())) + ".pt"

    def load_checkpoint(self):
        root = os.path.dirname(sys.modules['__main__'].__file__)
        file_names = glob.glob(os.path.join(root, self.checkpoint_folder, self.prefix + "*.pt"))
        if len(file_names) == 0:
            print("[!] Checkpoint not found.")
            return {}
        file_name = file_names[-1]  # Pick the most recent file.
        print("[+] Checkpoint Loaded. '{}'".format(file_name))
        return torch.load(file_name, map_location=self.device_name)

    def evaluate(self, batch_size):
        raise NotImplementedError


class RNNEvaluator(Evaluator):
    config = ConfigRNN.instance()

    def __init__(self, model):
        super().__init__(model)
        self.current_epoch = 0
        # self.model.eval()

        # Load model & optimizer.
        checkpoint = self.load_checkpoint()
        try:
            self.model.load_state_dict(checkpoint["model"])
        except KeyError:
            # There is no checkpoint
            pass

    def evaluate(self, review_vectors):
        with torch.no_grad():
            for review_vector in review_vectors:
                input_data = review_vector.to(torch.device(self.device_name))
                return self.model(input_data)


def prepare():
    config = ConfigRNN.instance()
    embed = Embed()
    embedding_model = embed.get_embedding_model()
    if config.EMBED_METHOD == "DEFAULT":
        model = ReviewParser()
    else:
        model = ReviewParser(pretrained=torch.from_numpy(embedding_model.wv.vectors).float())
    evaluator = RNNEvaluator(model)
    return evaluator, embed


def main():
    evaluator, embed = prepare()
    review_vector = embed.review2vec(sys.argv[1])
    print(evaluator.evaluate(review_vector))


def test():
    evaluator, embed = prepare()

    test_review_pos = "I love this movie. This is the best movie I've ever seen."
    review_vectors = embed.review2vec(test_review_pos)
    print(evaluator.evaluate(review_vectors=review_vectors))

    test_review_neg = "I hate this movie. This is the worst movie I've ever seen."
    review_vectors = embed.review2vec(test_review_neg)
    print(evaluator.evaluate(review_vectors=review_vectors))


if __name__ == "__main__":
    main()
