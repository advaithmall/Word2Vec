import torch
import pandas as pd
from collections import Counter
import re
import random
import pickle
from tqdm import tqdm
from dataset import Word2vecDataset
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import argparse
from model import Word2Vec_model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device: ", device)


def train(dataset, model, args):
    print("Training...")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    accur_list = list()
    for epoch in range(args.max_epochs):
        for batch, (context, target, sampling) in enumerate(dataloader):
            model.train()
            optimizer.zero_grad()
            y_pred = model(context, target).to(device)
            loss = loss_function(y_pred, sampling)
            loss.backward()
            optimizer.step()
            #calculate accuracy
            pred_list = list()
            for i in range(len(y_pred)):
                pred_list.append(y_pred[i].argmax().item())
            pred_list = pred_list
            sampling = sampling.tolist()
            acc = 0
            for i in range(len(pred_list)):
                if pred_list[i] == sampling[i]:
                    acc += 1
            acc = acc/len(pred_list)
            accur_list.append(acc)
            print({
                'epoch': epoch, 'batch': batch, 'loss': loss.item(), 'acc': acc})
    print("avg acc: ", sum(accur_list)/len(accur_list))


def eval(dataset, model, args):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loss_function = nn.CrossEntropyLoss()
    accur_list = list()
    for batch, (context, target, sampling) in enumerate(dataloader):
        y_pred = model(context, target).to(device)
        loss = loss_function(y_pred, sampling)
        #calculate accuracy
        pred_list = list()
        for i in range(len(y_pred)):
            pred_list.append(y_pred[i].argmax().item())
        pred_list = pred_list
        sampling = sampling.tolist()
        acc = 0
        for i in range(len(pred_list)):
            if pred_list[i] == sampling[i]:
                acc += 1
        acc = acc/len(pred_list)
        accur_list.append(acc)
        print({
            'batch': batch, 'loss': loss.item(), 'acc': acc})


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="train")
parser.add_argument('--max-epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument('--sequence-length', type=int, default=4)
args = parser.parse_args()


print("Making Dataset...")
data = Word2vecDataset()
print("Dataset made")
model = Word2Vec_model(len(data.word_set)).to(device)
l1 = int(0.80*len(data))
l2 = len(data) - l1
train_set, test_set = random_split(data, [l1, l2] )
train(train_set, model, args)
model.eval()
#save model
torch.save(model, "model.pt")
torch.save(train_set, "train_set.pt")
torch.save(test_set, "test_set.pt")
torch.save(data, "data.pt")


curr_layer = model.embedding.weight.data
curr_layer = curr_layer.cpu().numpy()
print(curr_layer.shape)
word_dict = data.index_to_word

with open("word_dict.pickle", "wb") as f:
    pickle.dump(word_dict, f)
with open("neural_embeddings.pickle", "wb") as f:
    pickle.dump(curr_layer, f)
