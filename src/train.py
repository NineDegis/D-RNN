from data import *
from model import *

# from tensorBoardLogger import *

config = ConfigManager("RnnImdb").load()
batch_size = int(config["BATCH_SIZE"])
acl_imdb = ACLIMDB(batch_size=batch_size, word_embedding='CBOW', is_eval=False, test_mode=True)

lstm = RnnImdb(torch.FloatTensor(acl_imdb.data.embedding_model.wv.vectors))
optimizer = torch.optim.SGD(lstm.parameters(), lr=0.001, weight_decay=0.0003)
criterian = nn.NLLLoss()


# TODO(hyungsun): Revive `Trainer` class.
def main():
    print("Start training!!")
    for i in range(100):
        sum_loss = 0
        # TODO(hyungsun): Consider `CUDA` while training.
        for batch_idx, (data, target) in enumerate(acl_imdb.load()):
            optimizer.zero_grad()
            input_data = data.view(-1, batch_size, 1)
            output, hidden, cell = lstm(input_data)
            loss = criterian(output, target)
            loss.backward()
            optimizer.step()
            sum_loss += loss
            print("loss : ", float(loss * 100))


if __name__ == "__main__":
    main()
    print("End")
