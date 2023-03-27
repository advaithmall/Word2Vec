import torch
from torch import nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class Word2Vec_model(nn.Module):
    def __init__(self, vocab_size):
        super(Word2Vec_model, self).__init__()
        self.embedding_dim = 300
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim).to(device)
        self.linear1 = nn.Linear(2*self.embedding_dim, 400).to(device)
        self.activation1 = nn.ReLU().to(device)
        self.linear2 = nn.Linear(400, 300).to(device)
        self.activation2 = nn.Tanh().to(device)
        self.linear3 = nn.Linear(300, 2).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, context, target):
        context = self.embedding(context).to(device)
        target = self.embedding(target).to(device)
        context = torch.sum(context, dim=1).to(device)
        x = torch.cat((context, target), dim=1).to(device)
        x = self.linear1(x).to(device)
        x = self.activation1(x).to(device)
        x = self.linear2(x).to(device)
        x = self.activation2(x).to(device)
        x = self.linear3(x).to(device)
        x = self.softmax(x).to(device)

        return x


