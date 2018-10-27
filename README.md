# D-RNN
RNN model incorporating Development process

## 1. Dataset
This dataset contains movie reviews along with their associated binary
sentiment polarity labels.  

### 1.1 Data organization  
The dataset contains 50,000 reviews split evenly into 25000 train
and 25000 test sets. Dataset also include an additional 50,000 unlabeled
documents for unsupervised learning.  

### 1.2 Download  
> http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz  

### 1.3 Publications Using the Dataset  
> http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf

## 2. Training

### 2.1 Using Tensorboard
```
./tbrun.sh
```
This shell script do things below.
1. Check if Tensorboard is running in the background.
2. If there is no running Tensorboard process, make Tensorboard run and `disown` it.
3. Runs `python triain.py` in the background, and `disown` it.  

### 2.2 Without Tensorboard
```
python train.py
```

### 2.3 Running Tensorboard without Training
```
./tbrun.sh no
```
