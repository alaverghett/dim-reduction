# Implementation of mds and isomap dimensionality reduction
# Both are run using the S shaped dataset from sklearn, and the results are plotted

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.spatial.distance as dist
import sklearn.manifold as sk
import sklearn as sklearn
from sklearn.decomposition import PCA
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import scipy.spatial.distance as d
import math


# Implementation of mds
def mds(D, n_components):
    P2 = np.square(D)
    J = np.identity(len(D)) - (1/len(D) * np.ones((len(D), len(D))))
    B = -0.5 * np.matmul(np.matmul(J, P2), J)
    values, vectors = np.linalg.eig(B)
    vectors = vectors.real
    X = vectors[:, 0:n_components]
    eigMat = np.zeros((n_components, n_components))
    for i in range(0, n_components):
        eigMat[i, i] = math.sqrt(values[i])
    
    X = np.matmul(X, eigMat)
    return X


# Implementation of isomap
def isomap(D, n_components, knn):
    g = nx.Graph()
    for i in range(0, len(D)):
        g.add_node(i)
    A = sklearn.neighbors.kneighbors_graph(D, n_neighbors=knn, include_self=False, mode='distance')
    B = A.toarray()
    i = 0
    j = 0
    for i in range(i, len(B)):
        for j in range(j, len(B)):
            if B[i][j] != 0:
                g.add_edge(i, j, weight=B[i][j])
        j = 0
    distMat = nx.floyd_warshall_numpy(g)
    return mds(distMat, n_components)


# The shapes created from mds and isomap look identical to those from sklearn.
# The code below graphs both of them using the S-shaped dataset

n_points = 500
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
n_neighbors=10
n_components=2
dist = sklearn.metrics.pairwise.pairwise_distances(X)
fit_mds = mds(dist, n_components)
fit_iso = isomap(X, n_components, n_neighbors)
fig=plt.figure(figsize=(15,8))
plt.suptitle("MDS",fontsize=20)
ax=fig.add_subplot(251,projection='3d')
ax.scatter(fit_mds[:,0],fit_mds[:,1], c=color, cmap=plt.cm.Spectral)
fig=plt.figure(figsize=(15,8))
plt.suptitle("ISOMAP",fontsize=20)
ax=fig.add_subplot(251,projection='3d')
ax.scatter(fit_iso[:,0],fit_iso[:,1], c=color, cmap=plt.cm.Spectral)
plt.show()