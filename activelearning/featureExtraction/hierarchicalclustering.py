import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import math
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn import preprocessing
from scipy.cluster import hierarchy


def cleandata(dataset, columnidx):
    # test = dataset.iloc[:, [16]].values
    # outlier = [1 for t in test if math.isnan(t)]
    # # len label: 5481
    # print("is known phrase: ", sum(outlier))
    f_knownphrase = dataset.iloc[:, [columnidx]].values
    for i in range(len(f_knownphrase)):
        if math.isnan(f_knownphrase[i]):
            f_knownphrase[i] = 0
    return f_knownphrase


def plotClusters(X, labels, n_clusters):
    colors = ['red', 'blue', 'green', 'orange', 'pink']
    for i in range(n_clusters):
        plt.scatter(X[labels == i, 6], X[labels == i, 7], s=50, marker='o', color=colors[i])

    plt.show()


def plotDendrogram(X, labels):
    linked = linkage(X, 'complete')
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               labels=labels,
               distance_sort='descending',
               show_leaf_counts=True)
    plt.show()
    plt.savefig("../../outputs/dendogram_3c.png")

def normalize(x):

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

def plotBySpipy(x):
    z = hierarchy.linkage(x, 'single')
    plt.figure()
    dn = hierarchy.dendrogram(z)
    hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    dn1 = hierarchy.dendrogram(z, ax=axes[0], above_threshold_color='y',
    orientation = 'top')
    dn2 = hierarchy.dendrogram(z, ax=axes[1],
    above_threshold_color = '#bcbddc',
    orientation = 'right')
    hierarchy.set_link_color_palette(None)
    plt.show()


def run():
    dataset = pd.read_csv('../../outputs/features_results_all.csv')
    f_knownphrase = cleandata(dataset, 16)
    X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]].values + f_knownphrase
    X = normalize(X)

    model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    model.fit(X)
    labels = model.labels_
    print("positive labels: ", sum(labels))
    print("negative labels: ", len(labels) - sum(labels))


    # plotClusters(X, labels, 5)
   #  plotDendrogram(X, labels)
    plotBySpipy(X)



if __name__ == '__main__':
    run()