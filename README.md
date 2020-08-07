# NLP BASED AUTHORSHIP CLUSTERING ON ENGLISH EDITORIAL DOCUMENTS

1) About the data
2) Reading the data
3) Preprocessing the data
4) Types of clustering
5) K-means clustering
6) Silhouette score
7) Selecting K-value by silhouette score
8) Bcubed F-score
9) Homogeneity , Completeness , V-measure
10) Scores of the model
11) Result

# About the data
We collected books from the five authors and collected 10 chapters from each book. These 50 files will form our data. The authors are
1)Chethan bhagath
2)Preethi Shenoy
3)Durjoy Datta
4)Sundeep Nagarkar
5)Vikram Chandra

# Reading the data
To read the data in this case we used a function called OS.walk().
OS.walk() generate the file names in a directory  by walking  either top-down or bottom-up. For each directory in the tree rooted at directory top (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames).
root : Prints out directories only from what you specified.
dirs : Prints out sub-directories from root.
files : Prints out all files from root and directories.

# Preprocessing the data
In preprocessing the data we will tokenize the files (i.e divides the file into the tokens). Tokens are the single words. It basically divides the large body of text into small lines or words. There are different types of tokenizers functions in nltk module.  Later, we  check the usefulness of the word in the document by checking the frequency. We will remove the unnecessary words. This can be done by tf-idf function. 
            In preprocessing the data we remove the unnecessary words(i.e is,the,and etc)and we only consider the useful words.to know the usefulness of a word we use a function called tf-idf(term frequency–inverse document frequency).
            Tf-idf  is a well know method to evaluate how important is a word in a document. tf-idf is a very interesting way to convert the textual representation of information into a vector space model(VSM). VSM is an algebraic model representing textual information as a vector, the components of this vector could represent the importance of a term (tf–idf).

# Types of clustering
1)K-means clustering
2)Hierarchical clustering
3) Minibatch K-means clustering
  1) K-means clustering- K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data. The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K.
  
  
 # Silhouette score
Silhouette score is measure of how close each point in a cluster is to the points in the neighbouring clusters. It’s a neat way to find out the optimum value for k during k-means clustering. Silhouette values lies in the range of [-1, 1]. A value of +1 indicates that the sample is far away from its neighboring cluster and very close to the cluster its assigned. Similarly, value of -1 indicates that the point is close to its neighboring cluster than to the cluster its assigned. And, a value of 0 means its at the boundary of the distance between the two cluster.

# Selecting K-value by silhouette score
Higher is the average silhouette score better is the configuration of the cluster. So we choose the K value which gives highest silhouette score.

# Bcubed F-score
Precision : The precision of a collection of documents is the average of the precisions of all mentions in all documents. The precision of a mention m is calculated as the number of mentions correctly predicted to be in the same cluster as m (including m) divided by the number of mentions in the predicted cluster containing m.
Recall : The recall of a collection of documents is the average of the recalls of all mentions in all documents. The recall of a mention m is calculated as the number of mentions correctly predicted to be in the same cluster as m (including m) divided by the number of mentions in the true cluster containing m.




In this model as we give priority to precision we use F1-score .Therefore,β = 0.5.

# Homogeneity , Completeness , V-measure

Homogeneity metric of a cluster labeling given a ground truth. A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class.

<img width="364" alt="Screenshot 2020-08-07 at 12 47 28 PM" src="https://user-images.githubusercontent.com/62896459/89620376-2fbbcc80-d8ad-11ea-9cea-df88fbbeee7f.png">


Completeness  metric of a cluster labeling given a ground truth. A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster. This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values won’t change the score value in any way

V-measure is an entropy-based measure which explicitly measures how successfully the criteria of homogeneity and completeness have been satisﬁed. Vmeasure is computed as the harmonic mean of distinct homogeneity and completeness scores, just as precision and recall are commonly combined into F-measure . As F-measure scores can be weighted, V-measure can be weighted to favor the contributions of homogeneity or completeness.

# Scores of our model
homogeneity_score	 0.867964899589081
completeness_score	 0.8108732905194428
v_measure_score	 0.8384483487830118
silhouette_score--eculidean	 0.3606266573120621
silhouette_score--cosine	 0.5544331052391969

# Result 








