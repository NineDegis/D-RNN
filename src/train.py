from data import *
from model import *

# from tensorBoardLogger import *

config = ConfigManager("RnnImdb").load()
batch_size = int(config["BATCH_SIZE"])
acl_imdb = ACLIMDB(batch_size=batch_size, word_embedding='CBOW', is_eval=False, test_mode=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lstm = RnnImdb(torch.FloatTensor(acl_imdb.data.embedding_model.wv.vectors))
optimizer = torch.optim.SGD(lstm.parameters(), lr=0.001, weight_decay=0.0003)
criterion = nn.NLLLoss()

# TODO(yongha, sejin): Revive `Trainer` class.
def main():
    print("Start training!!")
    for i in range(1000000):
        sum_loss = 0
        for batch_idx, (data, target) in enumerate(acl_imdb.load()):
            data, target = data.to(device=device), target.to(device=device)
            optimizer.zero_grad()
            input_data = data.view(-1, batch_size, 1)
            output, hidden, cell = lstm(input_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            sum_loss += loss
            print("epoch : ", i, "/", 999999)
            print("batch idx : ", batch_idx)
            print("loss : ", float(loss))
            print("target : ", target)
            print("output : ", output)
            print()

if __name__ == "__main__":
    main()
    print("End")
