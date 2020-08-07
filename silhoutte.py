
from sklearn.cluster import  KMeans
from sklearn.metrics import silhouette_score


def silhoutte(num_clusters, dist):
    sil_values = {}
    avg = 0
    for num_c in range(num_clusters, num_clusters * 2):
        # k_means_.euclidean_distances = new_euclidian_distances
        # km = SpectralClustering(n_clusters = num_clusters)
        # km = AgglomerativeClustering(n_clusters = num_clusters, linkage="ward" )
        # km= AffinityPropagation(preference=5).fit(dist)
        # km= MiniBatchKMeans(n_clusters=num_clusters, batch_size=50)
        km = KMeans(n_clusters=num_c, algorithm="auto", init="k-means++")
        clusters = km.fit(dist)
        # clusters = km.labels_.tolist()
        # clusters = km.fit_predict(tfidf_matrix)
        sil_avg = silhouette_score(dist, clusters.labels_)
        sil_values[sil_avg] = num_c
        avg += sil_avg
        print("Silhouette score  for n=", num_c, "is   ::: ", sil_avg)
    sil_avg = avg / (num_clusters);
    print("Average silhousse score is :", sil_avg)
    final_value = []
    for s in sil_values.keys():
        if (s >= sil_avg):
            final_value.append(s)
    return sil_values.get(min(final_value))
