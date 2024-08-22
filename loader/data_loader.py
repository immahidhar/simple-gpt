import torch

class DataLoader():
    def __init__(self, dataPath, gptConfig):
        self.dataPath = dataPath
        self.gptConfig = gptConfig
        self.data, self.train_data, self.val_data = None, None, None

    def loadData(self):
        # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
        with open(self.dataPath, 'r', encoding='utf-8') as f:
            self.data = f.read()

    def splitData(self, encode):
        # Train and test splits
        data = torch.tensor(encode(self.data), dtype=torch.long)
        n = int(0.9*len(data)) # first 90% will be train, rest val
        self.train_data = data[:n]
        self.val_data = data[n:]

    # data loading
    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.gptConfig.block_size, (self.gptConfig.batch_size,))
        x = torch.stack([data[i:i+self.gptConfig.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.gptConfig.block_size+1] for i in ix])
        x, y = x.to(self.gptConfig.device), y.to(self.gptConfig.device)
        return x, y