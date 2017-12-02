import pandas as pd
import pymysql
import pymysql.cursors
import math
import numpy as np
import networkx as nx
from infomap import infomap

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
				if result == 0:
					out_df[i][j] = (1- 0.0001)/0.0001
					out_df[j][i] = (1- 0.0001)/0.0001
				else:
					out_df[i][j] = (1-result)/result
					out_df[j][i] = (1-result)/result
			j += 1
		i += 1
	return out_df

def main():	
	print("called main")

	data = pd.read_csv("matNew.txt", sep=' ', header = None)
	data.pop(6918)
	#infomapWrapper = infomap.Infomap("-N5 --two-level -k --include-self-links")

	file = open("inputNoZero12.txt", 'w')
	file.write("# A network in link list format")
	i = 0
	while i < len(data):
		j = i
		while j < len(data[i]):
			if data[i][j] != 0:
				file.write(str(i) + " " + str(j) + " " + str(data[i][j]) + "\n")
			j += 1
		i += 1

	print("finished adding edge")
	'''
	infomapWrapper.run()

	tree = infomapWrapper.tree

	print("Found %d modules with codelength: %f" % (tree.numTopModules(), tree.codelength()))
	'''
	

if __name__ == "__main__":
    main()
