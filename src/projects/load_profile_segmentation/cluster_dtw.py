import os
import matplotlib as mpl
mpl.rcParams['backend'] = "qt4agg"
import pymssql
import datetime
import numpy as np
import random
from matplotlib import dates
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import csv
import collections
from sklearn import decomposition
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
from sklearn import neighbors
from sklearn.cluster import AgglomerativeClustering
from numpy.random import rand
from random import randint
import mpld3
import time
import pickle
import simplekml
import operator
import mlpy
plt.style.use('ggplot')


############################
# load data and dictionary #
############################
df = pickle.load(open("data/DataframeSummer2007-2014.p", "rb"))
FeederDict = pickle.load(open("data/FeederDictSummer2007-2014.p", "rb"))
# swap key and value pairs #
# feeder_dict = {v[0]: [k, v[1], v[2]] for k, v in FeederDict.items()}
# set data set length #
n = len(df)

############################
#    build the matrix      #
############################
data_matrix = df.ix[0:n, :-4].as_matrix()

############################
#   feature normalisation  #
############################
X_scaled = preprocessing.scale(data_matrix.T).T[0:n, :]
print('After row normalisation %s:' % (str(X_scaled.shape)))

############################
#    plot raw data set     #
############################
# plt.figure(0, facecolor='w')
# plt.plot(pd.date_range("00:00", "23:59", freq="30min"), X_scaled.T, color='k', alpha=0.02)
# plt.title('Raw data')
# plt.grid(True)


def k_means_clust(data, num_clust, num_iter):
    # centroids = random.sample(list(data), num_clust)
    # b = np.random.randint(0, data.shape[0], num_clust)
    b = np.random.permutation(data.shape[0])[:num_clust]
    print('Initialisation ids: %s' % str(b))
    centroids = data[b]
    conv = []
    meta = []
    for n in range(num_iter):
        print(n)
        assignments = {}
        # assign data points to clusters #
        for ind, i in enumerate(data):
            min_dist = float('inf')
            closest_clust = None
            for c_ind, j in enumerate(centroids):
                cur_dist = mlpy.dtw_std(i, j, dist_only=True)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_clust = c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []
        # recalculate centroids of clusters #
        # key = cluster number #
        for key, init in zip(assignments, data[b]):
            clust_sum = 0
            for k in assignments[key]:
                # k = number of records (rows of matrix) #
                clust_sum = clust_sum + data[k]
            centroids[key] = [m / len(assignments[key]) for m in clust_sum]
            conv.append(float(mlpy.dtw_std(init, centroids[key], dist_only=True)))
        meta.append(conv)
        conv = []
    return [centroids, assignments, meta]


iterations = 15
num_clusters = 5

[cen, ass, meta] = k_means_clust(X_scaled, num_clusters, iterations)


############
# plotting #
############

y = np.array(meta).T
labels = ['cluster%d' for i in range(num_clusters)]
fig = plt.figure(0, facecolor='w')
for y_arr, label in zip(y, labels):
    plt.plot(range(iterations), y_arr, label=label, marker='o')
plt.title('Centroid distance from initialisation')
plt.ylabel('DTW distance')
plt.xlabel('Iterations')
plt.xlim([0, iterations - 1])
plt.legend()

time = pd.date_range("00:00", "23:59", freq="30min")
fig = plt.figure(1, facecolor='w')
frame = plt.gca()
plt.plot(time, X_scaled.T, color='k', alpha=0.03)
plt.title('Data shape: %s. Unique feeders: %d.' % (str(X_scaled.shape), len(df['FeederName'].unique())))
frame.axes.get_yaxis().set_visible(False)
plt.gcf().autofmt_xdate()
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.grid(True)

fig = plt.figure(2, facecolor='w')
for i in range(num_clusters):
    frame = plt.gca()
    plt.plot(time, cen[i].T, label='Cluster: %d' % (i), linewidth=4)
    legend = plt.legend(loc='best', fancybox=True, framealpha=0.5)
    legend.draggable(True)
    frame.axes.get_yaxis().set_visible(False)
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.title('Clusters')
    plt.grid(True)

for i in range(num_clusters):
    fig = plt.figure(i + num_clusters - 2, facecolor='w')
    frame = plt.gca()
    plt.plot(time, X_scaled[ass[i]].T, color='k', alpha=0.03)
    plt.title('Cluster: %d' % (i))
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    frame.axes.get_yaxis().set_visible(False)
    plt.grid(True)
    plt.hold(True)
    plt.plot(time, cen[i].T, color='#0099FF', linewidth=5)
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    frame.axes.get_yaxis().set_visible(False)
    plt.grid(True)
    plt.hold(False)
plt.show()
