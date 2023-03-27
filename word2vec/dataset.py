import torch
import pandas as pd 
from collections import Counter
import re
import random
import pickle
from tqdm import tqdm
from pprint import pprint
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device: ", device)
random.seed(12322)
class Word2vecDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.corp_list = self.load_words()
        self.word_set = self.get_word_set()
        self.word_to_index = self.wrd2idx()
        self.index_to_word = self.idx2wrd()
        self.idxlist = self.convert_to_idx()
        self.vocab_size = len(self.word_set)
        self.word_sample, self.tag_sample, self.neg_pos = self.get_samples()
    
    def load_words(self):
        with open('neural_corp.pickle', 'rb') as f:
            corp_list = pickle.load(f)
        return corp_list
    def get_word_set(self):
        word_set = set(self.corp_list)
        word_set.add("<unk>")
        word_set.add("eos")
        word_set.add("<sym>")
        word_set.add("numhere")
        return word_set
    def wrd2idx(self):
        curr_dict = {}
        k = 0
        freq_dict = {}
        curr_dict["<unk>"] = k
        for i in self.corp_list:
            if i in freq_dict:
                freq_dict[i] += 1
            else:
                freq_dict[i] = 1
        for i in self.word_set:
            if i == "<unk>":
                continue
            if freq_dict[i] > 2:
                k += 1
                curr_dict[i] = k
        return curr_dict
    def idx2wrd(self):
        curr_dict = {}
        for i in self.word_to_index:
            curr_dict[self.word_to_index[i]] = i
        return curr_dict
    def convert_to_idx(self):
        idxlist = []
        for i in self.corp_list:
            idxlist.append(self.word_to_index.get(i, self.word_to_index["<unk>"]))
        return idxlist
    def get_samples(self):
        print("entered function ...")
        idxlist = self.idxlist
        word_sample = []
        tag_sample = []
        neg_pos = []
        window = 3
        ch = len(idxlist)-10
        for i in tqdm(range(window, len(idxlist)-window), total = len(idxlist)-2*window, desc = "Sampling"):
            #positive sampling
            curr_list = []
            for k in range (i-window, i):
                curr_list.append(idxlist[k])
            for k in range(i+1, i+window+1):
                curr_list.append(idxlist[k])
            target = idxlist[i]
            sampling_type = 0
            word_sample.append(curr_list)
            tag_sample.append(target)
            neg_pos.append(sampling_type)
            curr = random.randint(0, ch)
            while idxlist[curr] == target:
                curr = random.randint(0, ch)
            target = curr
            target = idxlist[target]
            #negative sampling 1
            sampling_type = 1
            word_sample.append(curr_list)
            tag_sample.append(target)
            neg_pos.append(sampling_type)
        return word_sample, tag_sample, neg_pos
    def __len__(self):
        return len(self.word_sample)
    def __getitem__(self, idx):
        return torch.tensor(self.word_sample[idx]).to(device), torch.tensor(self.tag_sample[idx]).to(device), torch.tensor(self.neg_pos[idx]).to(device)

    