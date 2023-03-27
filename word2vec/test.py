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

model = torch.load("model.pt")
train_set = torch.load("train_set.pt")
test_set = torch.load("test_set.pt")
data = torch.load("data.pt")

eval(test_set, model, args)