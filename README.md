# Word2Vec
An implementation of CBOW technique of Word2Vec using negative Sampling

### Directory Structory

```
.
├── data.pt
├── dataset.py
├── eval.py
├── model.pt
├── model.py
├── neural_corp.pickle
├── neural_embeddings.pickle
├── test.py
├── test_set.pt
├── train.py
├── train_set.pt
├── word2vec_images
│   ├── an.png
│   ├── awesome.png
│   ├── crucial.png
│   ├── father.png
│   ├── study.png
│   ├── thinking.png
│   └── titanic.png
└── word_dict.pickle

1 directory, 19 files

```

### In word2vec folder download: data.pt, model.pt, neural_corp.pickle, neural_embeddings.pickle, test_set.pt, train_set.pt, word_dict.pickle

### word_dict.pickle: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/EZa7W8A3-ZpEsSYf4XhsD_cBJA-SOmmLfyafZCpS733jiQ?e=JFtWLj

### model.pt: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/Ee2QSHEVl_5DoIZHHGIcdGgB1dzqw-0CCAEF-gUlmfL-4A?e=780NJ3

### neural_corp.pickle: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/EW0yORPEZTlNn0W--YkzrGoB0qAn2G_3vnlt6qZaek4gjQ?e=x93ySr

### neural_embeddings.pickle: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/EfvJ1kpau7BCh4dhhOcJossBer9CdAs5reLPJEO1Ji25rQ?e=b9tssh

### train_set.pt: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/EQdYPTovE_JGo2-pc-fMDX0BKULGzX3RSukiS_vcqPu2Sw?e=urqSzp

### test_set.pt: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/EUDRXTUDhmtOnQiwXObIFYkBudeaZUjtN504TNgYyl02Ew?e=naxGZd

### data.pt: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/EXDkzXZM_ZNChYcTKapkSwYBBKAaGPociCHBo9so6VkWlw?e=8pdHWi


## To run code related to word2vec, first:
```
cd word2vec

```
### to look at plot images presented in report:

```
cd word2vec_images

```

### To train model and to save embeddings, run:

```
python3 -W ignore train.py

```

### To evaluate performance on test set run:

```

python3 -W ignore test.py

```

### To get the 10 closest words to a given word and to plot them, run:

```

python3 -W ignore eval.py

enter word upon prompt

```
