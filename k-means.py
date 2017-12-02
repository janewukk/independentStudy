from pyjarowinkler import distance
from math import*
import nltk
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from sklearn.metrics import euclidean_distances
import gensim
import pymysql
import pymysql.cursors
from gensim import corpora, models
import math
from textblob import TextBlob as tb
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import requests
import json
import time
from scipy.spatial.distance import cdist
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.decomposition import PCA
import pandas as pd

def connect_to_database():
    options = {
        'user': "root",
        'passwd': "root",
        'db': "KnowBase",
        'cursorclass' : pymysql.cursors.DictCursor
    }
    db = pymysql.connect(**options)
    db.autocommit(True)
    return db


def idf_token_overlap(m1, m2, df_dict):
    entity1 = m1["entity"].split()
    entity2 = m2["entity"].split()
    listT = entity1
    listT.extend(entity2)
    intersection_word = set.intersection(set(entity1), set(entity2))
    union_word = set.union(set(listT))
    numerator = 0
    denominator = 0
    for word in intersection_word:
        numerator += 1 / math.log(1+df_dict[word])
    for word in union_word:
        denominator += 1 / math.log(1+df_dict[word])
    return (numerator / denominator)
    
def build_df_matrix(wordList, df_dict):
    out_df = []
    for i in range(len(wordList)):
        temp = []
        for j in range(len(wordList)):
            temp.append(-1)
        out_df.append(temp)

    i = 0
    while i < len(wordList):
        j = i
        while j < len(wordList):
            result = idf_token_overlap(wordList[i], wordList[j], df_dict)
            if i == j:
                out_df[i][i] = 0
            else:
                out_df[i][j] = np.exp(-result)
                out_df[j][i] = np.exp(-result)
            j += 1
        i += 1
    return out_df


db = connect_to_database()
cur = db.cursor()
string = "select b.freebase_id, b.entity, b.relation, b.value, b.link_am_score, b.link_scroe, b.base_id, "
string += "b.freebase_entity from NoiseEntity b"
cur.execute(string)
results = cur.fetchall()
groundTrue = {}
label = []
entity = []
entityDict = {}
y = []
i= 0
wordList = []
allWord = []
allEntity = []
wordDict = {}
idx = 0
for resu in results:
    if result["freebase_id"] not in groundTrue:
        groundTrue[result["freebase_id"]] = i
        label.append(result["freebase_entity"].lower())
        i = i + 1
    if result["entity"] not in entityDict:
        entityDict[result["entity"].lower()] = 1
    entity.append(result["entity"].lower())
    y.append(groundTrue[result["freebase_id"]])
    
    temp = {}
    temp["n"] = result["entity"].lower().split()
    list_A = result["relation"].lower().split()
    list_A.extend(result["value"].lower().split())
    temp["A"] = list_A
    temp["f_id"] = result["freebase_id"]
    temp["f_entity"] = result["freebase_entity"]
    temp["score"] = result["link_scroe"]
    temp["am_score"] = result["link_am_score"]
    temp["relation"] = result["relation"]
    temp["value"] = result["value"]
    temp["id"] = result["base_id"]
    temp["entity"] = result["entity"]
    wordList.append(temp)
    allWord.extend(result["entity"].lower().split())
    allWord.extend(result["relation"].lower().split())
    allWord.extend(result["value"].lower().split())
    allEntity.append(result["entity"])
    entityList = result["entity"].lower().split()
    for item in entityList:
        if item not in wordDict:
            wordDict[item] = idx
            idx += 1



df_dict = {}
contentDf = pd.read_csv("wordCount.txt", sep='\t', header=None)
for i in range(len(contentDf[0])):
    first = contentDf[0][i]
    second = contentDf[1][i]
    df_dict[first] = second
    
df_matrix = build_df_matrix(wordList, df_dict)
print("finish building df matrix")

# In[ ]:

distortions = []

ini = 110
K = []
while ini < 180:
    K.append(ini)
    ini += 2

for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df_matrix)
    kmeanModel.fit(df_matrix)
    print(k)
    distortions.append(sum(np.min(cdist(df_matrix, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / len(df_matrix[0]))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


