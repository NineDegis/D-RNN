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

    test_review_pos = """
    Bromwell High is nothing short of brilliant. Expertly scripted and perfectly delivered, this searing parody of a 
    students and teachers at a South London Public School leaves you literally rolling with laughter. It's vulgar, 
    provocative, witty and sharp. The characters are a superbly caricatured cross section of British society (or to be 
    more accurate, of any society). Following the escapades of Keisha, Latrina and Natella, our three 'protagonists' 
    for want of a better term, the show doesn't shy away from parodying every imaginable subject. Political correctness
    flies out the window in every episode. If you enjoy shows that aren't afraid to poke fun of every taboo subject 
    imaginable, then Bromwell High will not disappoint!
    """
    review_vectors = embed.review2vec(test_review_pos)
    print(evaluator.evaluate(review_vectors=review_vectors))

    test_review_neg = """
    Story of a man who has unnatural feelings for a pig. Starts out with a opening scene that is a terrific example of 
    absurd comedy. A formal orchestra audience is turned into an insane, violent mob by the crazy chantings of it's 
    singers. Unfortunately it stays absurd the WHOLE time with no general narrative eventually making it just too off 
    putting. Even those from the era should be turned off. The cryptic dialogue would make Shakespeare seem easy to a 
    third grader. On a technical level it's better than you might think with some good cinematography by future great 
    Vilmos Zsigmond. Future stars Sally Kirkland and Frederic Forrest can be seen briefly.
    """
    review_vectors = embed.review2vec(test_review_neg)
    print(evaluator.evaluate(review_vectors=review_vectors))


if __name__ == "__main__":
    test()
