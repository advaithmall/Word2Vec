import pprint
import json
import sys
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pickle
from scipy.linalg import svd
from sklearn.utils.extmath import randomized_svd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


with open("word_dict.pickle", "rb") as f:
    curr_dict = pickle.load(f)
with open("neural_embeddings.pickle", "rb") as f:
    U = pickle.load(f)

print("loaded data")
word2idx = {}
for key in curr_dict.keys():
    item = curr_dict[key]
    word2idx[item] = key

example = input("Enter word: ")
print("Getting similar words to: ", example)
titanic_vec = U[word2idx[example]]
sim_dict = {}
for key in word2idx.keys():
    temp_vec = U[word2idx[key]]
    prod = np.dot(titanic_vec, temp_vec)
    sim_dict[key] = prod
#sort dict based on values
sorted_sim_dict = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
curr_list = sorted_sim_dict[:10]
embed_sim_list = []
words = []
for item in curr_list:
    idx = word2idx[item[0]]
    words.append(item[0])
    embed_sim_list.append(U[idx])
X = np.array(embed_sim_list)
print("started plotting tsne graph...")
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8, 8))
#also plot for the current word
#add a plot title
plt.title("Similar words to: " + example)
for i in range(len(words)):
    col = 'blue'
    plt.scatter(X_tsne[i, 0], X_tsne[i, 1], marker='o',
                color=col, label="my embeddings")
    plt.text(X_tsne[i, 0]+0.02, X_tsne[i, 1]+0.02, words[i], fontsize=9)
print("Top 10 similar words to: ", example)
print(words)
plt.show()
