import networkx as nx
import community
import pandas as pd
import matplotlib.pyplot as plt
from random import randint, random
import pymysql
import pymysql.cursors
import math
import numpy as np
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

def get_graph(mat):
	G = nx.Graph()
	row = 0
	while row < len(mat):
		col = row
		while col < len(mat[row]):
			G.add_edge(str(mat[row][0]), str(mat[row][col]))
			col += 1
		row += 1
	return G

def get_community_list(comm_dict):
	num_communities = 0
	for key, value in comm_dict.items():
		if value > num_communities:
			num_communities = value
	print("There are in total", num_communities, "communities")

	communities = [[] for i in range(num_communities + 1)]

	for key, value in comm_dict.items():
		communities[value].append(key)

	for i in range(len(communities)):
		print("Size of community", i, "is", len(communities[i]))

	return communities

def get_community_sample(community, size, full=False):
	# Returns a random sample of size, size from the specified community. If full is true
	# it just returns the full community.

	community_size = len(community)
	if full or community_size < size:
		return community
	used_indices = set()
	
	samples=[]

	i = 0
	while i < size:
		index = randint(0, community_size-1)
		if index not in used_indices:
			used_indices.add(index)
			samples.append(community[index])
			i += 1
	return samples

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



G = get_graph(df_matrix)
print("Finished creating graph")
p = community.best_partition(G, weight='weight')
print("Finished calculating clusters")

communities = get_community_list(p)

comm_nodes = [[] for i in range(len(communities))]
for i in range(len(comm_nodes)):
	comm_nodes[i] = get_community_sample(communities[i], 400)

comm_nodes_list = [item for sublist in comm_nodes for item in sublist]

subgraph = G.subgraph(comm_nodes_list)

pos=nx.spring_layout(subgraph)
colors = [(random(), random(), random()) for _i in range(len(communities))]

# colors=['blue', 'red', 'orange', 'yellow', 'green', 'cyan', 'black', '']
for i in range(len(communities)):
	nx.draw_networkx_nodes(subgraph, pos, nodelist=comm_nodes[i], node_color=colors[i])

edges = subgraph.edges()
nx.draw_networkx_edges(G,pos, edgelist=edges)
plt.show()