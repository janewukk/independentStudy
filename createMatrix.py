from pyjarowinkler import distance
from math import*
import nltk
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


# In[106]:

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


# data clean, exclude stop word, need to to lower
def exclude_stop_word(bloblist):
    stop = set(stopwords.words('english'))
    filtered_words = [i for i in bloblist[0].lower().split() if i not in stop]
    return filtered_words


def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)


def string_similarity(m1, m2):
    entity1 = m1["n"]
    entity2 = m2["n"]
    str1 = " ".join(entity1)
    str2 = " ".join(entity2)
    return distance.get_jaro_distance(str1, str2, winkler=True, scaling=0.1)


def build_matrix_string_similarity(wordList):
    out = []
    for i in range(len(wordList)):
        temp = []
        for j in range(len(wordList)):
            temp.append(-1)
        out.append(temp)

    i = 0
    while i < len(wordList):
        j = i
        while j < len(wordList):
            result = string_similarity(wordList[i], wordList[j])
            if i == j:
                out[i][i] = 1
            else:
                if result < 0.6: 
                    result = 0

                out[i][j] = result
                out[j][i] = result
            j += 1
        i += 1

    file = open("matNew.txt", 'w')
    for line in out:
        for word in line:
            file.write(str(word) + " ")
        file.write("\n")

    return out


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
file_t = open("data.txt", 'w')
for result in results:
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
    wordList.append(temp)


string_similarity_matrix = build_matrix_string_similarity(wordList)
print("finish building")



