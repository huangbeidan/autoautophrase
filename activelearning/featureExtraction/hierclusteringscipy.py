import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import diff
from sklearn.cluster import AgglomerativeClustering
import math
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn import preprocessing
from scipy.cluster import hierarchy

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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


def normalize(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df


def getdata():
    dataset = pd.read_csv('../../outputs/features_results_all.csv')
    f_knownphrase = cleandata(dataset, 16)
    # X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]].values + f_knownphrase
    X = dataset.iloc[:, [0, 1, 2, 3, 4, 6, 7]].values + f_knownphrase
    X = normalize(X)
    return X



def featureDistribution():
    X = getdata()
    fig = plt.figure(figsize=(20, 40), facecolor='white')
    plot_number = 1
    for (columnName, columnData) in X.iteritems():
        axes = plt.subplot(6, 3, plot_number)
        bin = len(np.unique(columnData)) // 2
        if bin==0:
            bin = 10
        columnData.hist(ax=axes, label=columnName, bins=40 )
        axes.set_title(columnName)
        plot_number += 1
    plt.tight_layout()
    plt.show()
    fig.savefig('../../outputs/features_hist.png')
    # fig, axes = plt.subplots(nrows=4, ncols=2)
    # fig.subplots_adjust(hspace=0.5)
    # fig.suptitle('Distributions of Features')
    # feature_names = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    #
    # for ax, feature, name in zip(axes.flatten(), X, feature_names):
    #     sns.distplot(feature, ax=ax, bins=len(np.unique(X[0])) // 2)
    #     ax.set(title=name[:-4].upper(), xlabel='cm')



def getLabels(method = 'ward'):
    X = getdata()
    Z = linkage(X.to_numpy(), method)
    # c, coph_dists = cophenet(Z, pdist(X))
    knee = np.diff(Z[::-1, 2], 2)
    num_clust1 = knee.argmax() + 2
    print("num_clust1: ", num_clust1)
    # knee[knee.argmax()] = 0
    # num_clust2 = knee.argmax() + 2

    cuttree = hierarchy.cut_tree(Z, n_clusters=num_clust1+3)
    clustered_labels = cuttree.flatten()

    # create a dictionary to hold the information
    output = {}
    for i in range(len(clustered_labels)):
        output[i] = clustered_labels[i]
    return output


def plot(X):

    Z = linkage(X.to_numpy(), 'single')
    Z1 = linkage(X.to_numpy(), 'complete')
    Z2 = linkage(X.to_numpy(), 'average')
    Z3 = linkage(X.to_numpy(), 'weighted')
    Z4 = linkage(X.to_numpy(), 'centroid')
    Z5 = linkage(X.to_numpy(), 'median')
    Z6 = linkage(X.to_numpy(), 'ward')
    c, coph_dists = cophenet(Z, pdist(X))

    # plt.figure(figsize=(25, 10))
    # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('sample index')
    # plt.ylabel('distance')
    # dendrogram(
    #     Z,
    #     leaf_rotation=90.,  # rotates the x axis labels
    #     leaf_font_size=8.,  # font size for the x axis labels
    # )
    # plt.show()
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    # plt.title('Hierarchical Clustering Dendrogram (truncated)')
    # plt.xlabel('sample index')
    # plt.ylabel('distance')

    axes[0, 0].set_title('simple')
    dendrogram(
        Z,
        ax = axes[0,0],
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )

    axes[0, 1].set_title('complete')
    dendrogram(
        Z1,
        ax=axes[0,1],
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    axes[1, 0].set_title('average')
    dendrogram(
        Z2,
        ax=axes[1,0],
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    axes[1, 1].set_title('weighted')
    dendrogram(
        Z3,
        ax=axes[1,1],
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    axes[2, 0].set_title('centroid')
    dendrogram(
        Z4,
        ax=axes[2,0],
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    axes[2, 1].set_title('median')
    dendrogram(
        Z5,
        ax=axes[2,1],
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )

    axes[3, 0].set_title('ward')
    dendrogram(
        Z5,
        ax=axes[3,0],
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )

    plt.show()

def run():
    dataset = pd.read_csv('../../outputs/features_results_all.csv')
    f_knownphrase = cleandata(dataset, 16)
    X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]].values + f_knownphrase
    X = normalize(X)
    # print(len(X))
    # plot(X)

    # Z = linkage(X.to_numpy(), 'single')
    # Z1 = linkage(X.to_numpy(), 'complete')
    # Z2 = linkage(X.to_numpy(), 'average')
    # Z3 = linkage(X.to_numpy(), 'weighted')
    # Z4 = linkage(X.to_numpy(), 'centroid')
    # Z5 = linkage(X.to_numpy(), 'median')
    # Z6 = linkage(X.to_numpy(), 'ward')
    # c, coph_dists = cophenet(Z, pdist(X))

    # knee1 = diff(Z[::-1, 2], 2)
    # num_clust = knee1.argmax() + 2
    # print("num_clust: ", num_clust)
    # print(hierarchy.leaves_list(Z))
    # print(hierarchy.fcluster(Z, 0.3))


if __name__ == '__main__':
    # getLabels()
    featureDistribution()
