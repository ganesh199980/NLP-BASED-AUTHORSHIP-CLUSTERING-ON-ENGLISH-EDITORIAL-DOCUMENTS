from __future__ import print_function
import pandas as pd
import nltk
import re
import os
import codecs
import numpy
import tokenizer

from sklearn.cluster.k_means_ import KMeans


dir = "cl"

authors = []
titles = []
synopses = []
doc = {}

j = 0
# ik = 0
# In[2]
for root, subfolders, files in os.walk(dir):
    # adding authors to list
    ti = []
    author = root.split("/")[-1]
    # unknown "" error. have to check once again

    if author == "":
        continue
    authors.append(author)

    for i in files:
        # adding article names to the list
        ti.append(i)
        if (i.startswith("unk")):
            titles.append("UNKNOWN")
        else:
            titles.append(author)
        print(root + "/" + i + "\n")
        f = codecs.open(root + "/" + i, "r", encoding="utf-8", errors='ignore')
        str = f.read()
        synopses.append(str.replace('.',''))
        # j += 1
        # doc[authors[-1]] = list(doc[[authors[-1]]].append(i))
    doc[author] = ti

print(authors)
print(titles[:2])

tt = []

ranks = []
for i in range(1, len(titles) + 1):
    ranks.append(i)

import tokenizer as tokenizer
# In[5]
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer( max_features=300
                             , use_idf=True, tokenizer=tokenizer.tokenize_only
                             , ngram_range=(1,3), analyzer="word")
tfidf_matrix = tfidf_vect.fit_transform(synopses)
print(tfidf_matrix[0,0])

from sklearn.metrics.pairwise import cosine_distances


def new_euclidian_distances(X, Y=None, Y_norm_squared=None, squared=False):
    return cosine_distances(X, Y)


from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
print
print

# In[7]


from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering

num_clusters = 4

import  silhoutte as sil


num_clusters = sil.silhoutte(num_clusters, dist)
num_clusters = 4
print("num clusters:: ",num_clusters)
km = KMeans(n_clusters=num_clusters)

clusters = km.fit(dist)

clusters = clusters.labels_.tolist()
# In[8]
films = {'title': titles, 'rank': ranks, 'synopses': synopses, 'cluster': clusters
    , 'authors': authors}
frame = pd.DataFrame(films, index=[clusters], columns=['rank', 'title', 'cluster', 'author'])
# print(frame)

frame['cluster'].value_counts()

grouped = frame['rank'].groupby(frame['cluster'])
print(grouped.mean())

# In[9]

print('top terms per cluster')

clust_details=[]
# order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print('cluster %d titles:\n' % i, end='')
    cou={}
    for title in frame.ix[i]['title'].values.tolist():
        cou[title]=0
    print('cluster  length:   %d\n' % len(frame.ix[i]['title'].values.tolist()), end='')
    for title in frame.ix[i]['title'].values.tolist():
        cou[title]+=1
        print(' %s, ' % title, end='')
    print('\n')
    print(cou)
    p = max(cou.values())/sum(cou.values())
    r = max(cou.values())/50
    print("Precision = ", p, " Recall = ", r , "F-Score = ", ((2*p*r)/(p+r)))
    clust_details.append(cou)
    print()
print()
print()

#print(clust_details)


def score():
    global F
    F = 0
    for cdict in clust_details:
        N1 = max(cdict.values())
        N2 = sum(cdict.values())
        N3 = 50
        P = N1 / N2
        R = N1 / N3
        F += (2 * P * R) / (P + R)


score()

print("BCubed F-Score = ", (F/num_clusters))


# In[10]
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

MDS()

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)

xs, ys = pos[:, 0], pos[:, 1]

print()
print()

cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'
    , 5: '#00FFFF', 6: '#FFE4C4', 7: '#000000', 8: '#626eb5', 9:"#ffeaeb"}
cluster_names = {0: 'Auth1',
                 1: 'Auth2',
                 2: 'Auth3',
                 3: 'Auth4',
                 4: 'Auth5',
                 5: 'Auth6',
                 6: 'Auth7',
                 7: 'Auth8',
                 8: 'Auth9',
                 9: 'Auth10'}
# In[11]
# some ipython magic to show the matplotlib plots inline
# inline

# create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

# group by cluster
groups = df.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(17, 9))  # set size
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

# iterate through groups to layer the plot
# note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params( \
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params( \
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelleft='off')

# ax.legend(numpoints=1)  # show legend with only 1 point

# add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=6)

plt.show()  # show the plot