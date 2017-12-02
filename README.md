# independentStudy

Agglomerative Clustering.ipynb):
This file is Hierarchical clustering based on different similarity methods: **string similarity** & string identity, input into Hierarchical clustering is affinity matrix
-- ignore the last part output

work-entity-idf.ipynb:
Three other similarity function: work-overlap, entity-overlap, idf-overlap

Louvain.py:
Just run louvain algorithm on graph input(networkx)

Research.ipynb
This file is Hierarchical clustering use sklearn-defined euclidean distance/cosine/ and their link method complete/average/ward, input is features

dbscan.ipynb
DBSCAN method with normalized scaled distance input: example()
   1   2    3 
1  0   99   8.2
2  99  0    15
3  8.2 15   0
eps choose from 0.1-3
min-sample = 5(ground true), this value does not matter that much in this case
dbscan output #cluster under different hyperparameter combination, and based on #cluster(output by dbscan), apply clustering algorithm which need #cluster as hyperparameter and output precision and recall(macro, micro, pairwise)
Find certain range that maximize both precision and recall (for string similarity, eps = 1.5-1.7), and calculate the weight threshold based on this

Then apply infomap, input is just string similarity result. (runInfomap.py will create the link file needed by infomap)

CreateMatrix.py
create distance matrix based on stringSimilarity method

