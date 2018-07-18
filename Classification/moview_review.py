# -*- coding: utf-8 -*-
"""
Created on Fri May  4 22:09:47 2018

@author: TP550LB
"""
import nltk.data
import pandas as pd
import numpy as np
import nltk, re,logging
from nltk.corpus import stopwords
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import namedtuple
from bs4 import BeautifulSoup
from nltk.stem import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.linear_model import SGDClassifier as sgd
from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import VotingClassifier as vc

train = pd.read_csv('labeledTrainData.tsv',delimiter="\t",header =0,quoting =3)
test = pd.read_csv('testData.tsv',delimiter="\t",header =0,quoting =3)
unlabeled_train = pd.read_csv('unlabeledTrainData.tsv',delimiter="\t",header=0,quoting =3)

''' cleaning text'''
def clean_text(review,remove_stopwords=True,stem_words=True):
    review_text = BeautifulSoup(review ,"lxml").get_text()
    
    text = review.lower().split()
    if remove_stopwords:
        stop = set(stopwords.words("english"))
        text = [w for w in text if not w in stop]
    review_text = " ".join(text)
    review_text = re.sub(r"[^A-Za-z0-9!?\'\`]", " ", review_text)
    review_text = re.sub(r"it's", " it is", review_text)
    review_text = re.sub(r"that's", " that is", review_text)
    review_text = re.sub(r"\'s", " 's", review_text)
    review_text = re.sub(r"\'ve", " have", review_text)
    review_text = re.sub(r"won't", " will not", review_text)
    review_text = re.sub(r"don't", " do not", review_text)
    review_text = re.sub(r"can't", " can not", review_text)
    review_text = re.sub(r"cannot", " can not", review_text)
    review_text = re.sub(r"n\'t", " n\'t", review_text)
    review_text = re.sub(r"\'re", " are", review_text)
    review_text = re.sub(r"\'d", " would", review_text)
    review_text = re.sub(r"\'ll", " will", review_text)
    review_text = re.sub(r"!", " ! ", review_text)
    review_text = re.sub(r"\?", " ? ", review_text)
    review_text = re.sub(r"\s{2,}", " ", review_text)
    
    if stem_words:
        words = review_text.lower().split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in words]
        review_text = " ".join(stemmed_words)
    return review_text.split()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def sentence(review, tokenizer):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences =[]
    for sen in raw_sentences:
        if len(sen)>0:
            sentences.append(clean_text(sen))
    return sentences
sentences = []
print("Parsing sentences from training data....\n")
for s in train["review"]:
    sentences+= sentence(s,tokenizer)
    
print("Parsing sentences from unlabelled data....\n")
for s in unlabeled_train["review"]:
    sentences+= sentence(s,tokenizer)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO )    

num_features = 300      # Word vector dimensionality                      
min_word_count = 5      # Minimum word count                        
num_workers = 1         # Number of threads to run in parallel
context = 20            # Context window size                                                                                    
downsampling = 1e-4     # Downsample setting for frequent words
from gensim.models import word2vec
print("Training Model....\n")
model = word2vec.Word2Vec(sentences, size = num_features,workers = num_workers, min_count = min_word_count, window = context, sample = downsampling)

model.init_sims(replace = True)
model_name = "Movie_Review"
model.save(model_name)

word_vectors = model.wv.syn0
num_clusters = int(word_vectors.shape[0]/5)

kmean = KMeans(n_clusters = num_clusters, n_init =5, verbose  =2)
idx = kmean.fit_predict(word_vectors)
word_centroid_map = dict(zip(model.wv.index2word, idx))
clean_train_reviews =[]
for r in train["review"]:
    clean_train_reviews.append(clean_text(r))
print("Training reviews are clean\n")
clean_test_reviews =[]
for r in test["review"]:
    clean_test_reviews.append(clean_text(r))
print("Test Reviews are clean")

def bag_of_centroids(wordlist,word_centroid_map):
    num_centroids = max(word_centroid_map.values())+1
    bag_of_centroids = np.zeros(num_centroids,dtype="float32")
    for w in wordlist:
        if w in word_centroid_map:
            bag_of_centroids[word_centroid_map[w]] +=1
    return bag_of_centroids

train_centroids1 = np.zeros((train["review"].size,num_clusters),dtype="float32")
i=0
for r in clean_train_reviews:
    train_centroids1[i] = bag_of_centroids(r,word_centroid_map)
    i+=1
    print(i)
test_centroids = np.zeros((test["review"].size,num_clusters),dtype="float32")
i=0
for r in clean_test_reviews:
    test_centroids[i] = bag_of_centroids(r,word_centroid_map)
    i+=1
def use_GridSearch(model, model_paramters, x_values):
    '''Find the optimal parameters for a model'''
    grid = GridSearchCV(model, model_paramters, scoring = 'roc_auc')
    grid.fit(x_values, train.sentiment)

    print("Best grid score = ", grid.best_score_)
    print("Best Parameters = ", grid.best_params_)

rfc_parameters = {'n_estimators':[100,200,300],'max_depth':[3,5,7,None],'min_samples_leaf': [1,2,3]}
rfc_model = rfc()
use_GridSearch(rfc_model, rfc_parameters, train_centroids)

# Logistic Regression
lr_parameters = {'C':[0.005,0.01,0.05],'max_iter':[4,5,6],'fit_intercept': [True]}
lr_model = lr()
use_GridSearch(lr_model, lr_parameters, train_centroids1)

sgd_parameters = {'loss': ['log'],'penalty': ['l1','l2','none']}
sgd_model = sgd()
use_GridSearch(sgd_model, sgd_parameters, train_centroids1)

def use_model(model, x_values):
    scores = cross_val_score(model, x_values, train.sentiment, cv = 5, scoring = 'roc_auc')
    model.fit(x_values, train.sentiment)
    mean_score = round(np.mean(scores) * 100,2) 
    print(scores)
    print()
    print("Mean score = {}".format(mean_score))
    
rfc_model = rfc(n_estimators = 300,max_depth = None, min_samples_leaf = 1)
use_model(rfc_model, train_centroids1)

lr_model = lr(C = 0.01, max_iter = 5,fit_intercept = True)
use_model(lr_model, train_centroids1)

sgd_model = sgd(loss = 'log',penalty = 'l1')
use_model(sgd_model, train_centroids1)

rfc_result = rfc_model.predict(test_centroids)
lr_result = lr_model.predict(test_centroids)
sgd_result = sgd_model.predict(test_centroids)

avg_result = (lr_result + rfc_result + sgd_result) / 3

avg_result_final = []
for result in avg_result:
    if result > 0.5:
        avg_result_final.append(1)
    else:
        avg_result_final.append(0)
        
avg_output = pd.DataFrame(data={"id":test["id"], "sentiment":avg_result_final})
avg_output.to_csv("avg_centroids_submission.csv", index=False, quoting=3)