# Projects: test_results
#
# :Author: Jaco du Toit <jacowp357@gmail.com>
# :Date: 23/09/2015
# :Description: Segementation results.
#
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
# import colors as kml_colors
# import operator
import clusterinitialisation as clusterinit
plt.style.use('ggplot')


############################
# load data and dictionary #
############################
# df = pickle.load(open('data/DataframeSummer2007-2014.p', "rb" ))
# FeederDict = pickle.load(open('data/FeederDictSummer2007-2014.p', 'rb'))
# season = 'winter'
# # swap key and value pairs #
# feeder_dict = {v[0]: [k, v[1], v[2]] for k, v in FeederDict.items()}
# # set data set length #
# n = len(df)

# ############################
# #    build the matrix      #
# ############################
# data_matrix = df.ix[0:n,:-4].as_matrix()

# ############################
# #   feature normalisation  #
# ############################
# X_scaled = np.trunc(preprocessing.scale(data_matrix.T).T[0:n, :] + 100).astype(int)
# print('After row normalisation %s:' % (str(X_scaled.shape)))

############################
#    plot raw data set     #
############################
# plt.figure(0, facecolor='w')
# plt.plot(pd.date_range("00:00", "23:59", freq="30min"), X_scaled.T, color='k', alpha=0.02)
# plt.title('Raw data')
# plt.grid(True)
# plt.show()

# y_norm = []
# y_nubs = []

# for perc in range(101):
# 	p = int(len(X_scaled)*(perc/float(100))) + 6000
# 	if p > n:
# 		break
# 	data = X_scaled[0:p, :]
# 	for i in range(5):
# 		start = time.clock()
# 		km = KMeans(5)
# 		km.fit(data)
# 		first = time.clock() - start
# 		y_norm.append((p, first))
# 		Centroids = clusterinit.GetCentroids().nubs(data, 5)
# 		start = time.clock()
# 		km = KMeans(5, init=np.array(Centroids), n_init=1)
# 		km.fit(data)
# 		second = time.clock() - start
# 		y_nubs.append((p, second))
# 		print('K-means done in %.4f seconds, on data shape %s, clusters = %d' % (first, str(data.shape), 5))
# 		print('K-means nubs done in %.4f seconds, on data shape %s, clusters = %d' % (second, str(data.shape), 5))

# pickle.dump(y_norm, open('norm_test.p', "wb"))
# pickle.dump(y_nubs, open('nubs_test.p', "wb"))

y_norm = pickle.load(open('norm_test.p', "rb"))
y_nubs = pickle.load(open('nubs_test.p', "rb"))

norm = pickle.load(open('norm.p', "rb"))
nubs = pickle.load(open('nubs.p', "rb"))

fig = plt.figure(1, facecolor='w')
plt.scatter([x[0] for x in y_norm], [y[1] for y in y_norm], marker='x', color='#FF6600', alpha=0.9)
plt.hold(True)
plt.scatter([x[0] for x in norm], [y[1] for y in norm], marker='x', label='Uniform initialisation', color='#FF6600', alpha=0.9)
plt.legend(loc=0, scatterpoints=1)
plt.grid(True)
plt.scatter([x[0] for x in y_nubs], [y[1] for y in y_nubs], marker='+', label='NUBS initialisation', color='#0066FF', alpha=0.9)
plt.legend(loc=0, scatterpoints=1)
plt.scatter([x[0] for x in nubs], [y[1] for y in nubs], marker='+', label='non-uniform binary split', color='#0066FF', alpha=0.9)
plt.xlabel('Observations')
plt.ylabel('Time (seconds)')
plt.axis([0, 122000, 0, 60])
plt.title('K-means initialisation comparison')
plt.show()

























############################
# dimensionality reduction #
############################
# p_components = 5
# start = time.clock()
# pca = decomposition.PCA(n_components=p_components, whiten=False)
# pca.fit(X_scaled)
# X_scores = pca.transform(X_scaled)
# print('PCA done in %.4f seconds...' % (time.clock() - start))
# print('Reduced dimensions: %s' % (str(X_scores.shape)))
# print('Explained variance: %s' % (str(pca.explained_variance_ratio_)))

###########################
#   plot pca components   #
###########################
# fig = plt.figure(1, facecolor='w')
# for c in range(p_components):
#     plt.plot(pca.components_[c,:], label='Component %d (var %.2f)' % ((c + 1), pca.explained_variance_ratio_[c]))
# plt.title('Principal components')
# plt.grid(True)
# plt.legend()
# plt.show()


################################
#     k-means centroids        #
# for non-uniform binary split #
################################
# number_clusters = 5
# Centroids = clusterinit.GetCentroids().nubs(X_scaled, number_clusters)
# fig = plt.figure(1, facecolor='w')
# plt.plot(np.array(Centroids).T, linewidth=2)
# plt.show()

###########################
#   k-means clustering    #
###########################
# start = time.clock()
# number_clusters = 5
# # km = KMeans(number_clusters)
# km = KMeans(number_clusters, init=np.array(Centroids), n_init=1)
# km.fit(X_scaled)
# # km.fit(X_scores)
# print('K-means done in %.4f seconds...' % (time.clock() - start))

# # reconstructed_centroids = pca.inverse_transform(km.cluster_centers_)
# reconstructed_centroids = km.cluster_centers_

# for i, centroid in enumerate(reconstructed_centroids):
#     fig = plt.figure(2 + i, facecolor='w')
#     indices = [j for j, x in enumerate(km.labels_.tolist()) if x == i]
#     plt.plot(X_scaled[indices].T, color='k', alpha=0.02)
#     plt.hold(True)
#     plt.plot(centroid, label='Centroid %d' % (i), linewidth=3, color='#0099FF', alpha=0.9)
#     plt.title('%s Centroid' % (season))
#     plt.grid(True)
#     plt.legend()
#     plt.hold(False)
#     plt.savefig('profile_%d.png' % (i))
# # plt.show()

# # add cluster index to dataframe #
# df['Clusters'] = km.labels_


######################################
# plot in browser with feeder labels #
######################################
# add cluster index to dataframe #
# df['Clusters'] = km.labels_
# figb, axb = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'), figsize=(12,10))
# scatter = axb.scatter(X_scores[:, 0], X_scores[:, 1], c=km.labels_, alpha=0.8, cmap=plt.cm.jet)
# axb.grid(color='white', linestyle='solid')
# axb.set_title("Scatter", size=20)
# tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=df['FeederName'].values.tolist())
# mpld3.plugins.connect(figb, tooltip)
# mpld3.show()

#############################################
# 3d scatter plot for PCA with 3 components #
#############################################
# fig = plt.figure(7, facecolor='w')
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_scores[:, 0], X_scores[:, 1], X_scores[:, 2], c=km.labels_)
# plt.title('3d PCA scores')
# plt.show()

########################
# plot to google earth #
########################
# color_list = ['yellow', 'blue', 'red', 'orange', 'green']
# kml = simplekml.Kml()
# multipnt = kml.newmultigeometry(name="Clusters")

# for i, item in enumerate(df['FeederName'].unique()):
#     temp_dict = dict(collections.Counter(df.loc[df['FeederName'] == item]['Clusters'].values))
#     clu = max(temp_dict.items(), key=operator.itemgetter(1))[0]
#     Lon = feeder_dict[item][1]
#     Lat = feeder_dict[item][2]
#     if Lon != 'NULL' and Lat != 'NULL':
#         # change GPS format - remove n/s/e/w and replace with signs #
#         if 'e' in Lon.lower():
#             Lon = '+' + str(Lon[:-1])
#         if 'w' in Lon.lower():
#             Lon = '-' + str(Lon[:-1])
#         if 's' in Lat.lower():
#             Lat = '-' + str(Lat[:-1])
#         if 'n' in Lat.lower() and Lat != 'NULL':
#             Lat = '+' + str(Lat[:-1])
#         pnt = kml.newpoint(coords=[(Lon, Lat)])
#         pnt.style.iconstyle.color = kml_colors.color_dict[color_list[clu]]
#         pnt.style.iconstyle.scale = 0.9
#         cluster = ''
#         for k, v in temp_dict.items():
#             cluster = cluster + ('Cluster %d: %.4f %% \n' % (k, float(v/len(df)*100)))
#         picpath = 'profile_%d.png' % (clu)
#         pnt.description = 'Feeder Description: %s \n\n Cluster: %d \n\n %s \n\n %s' % (str(item), clu, cluster, '<img src="' + picpath +'" alt="picture" width="380" height="260" align="left" />')
# kml.save("%s_clusters.kml" % (season))
# os.system("%s_clusters.kml" % (season))


