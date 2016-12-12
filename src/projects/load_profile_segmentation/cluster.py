# Projects: load_profile_segmentation
#
# :Author: Jaco du Toit <jacowp357@gmail.com>
# :Date: 23/09/2015
# :Description: Segementation of daily energy load profiles using k-means
#               clustering with different similarity metrics and uniform
#               binary split centroid initialisation.
#
import os
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
import colors as kml_colors
import operator
import clusterinitialisation as clusterinit
import matplotlib as mpl
mpl.rcParams['backend'] = "qt4agg"
plt.style.use('ggplot')

############################
# load data and dictionary #
############################
df = pickle.load(open("data/DataframeSummer2007-2014.p", "rb"))
FeederDict = pickle.load(open("data/FeederDictSummer2007-2014.p", "rb"))
season = "Summer"
# swap key and value pairs #
feeder_dict = {v[0]: [k, v[1], v[2]] for k, v in FeederDict.items()}
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
plt.figure(0, facecolor='w')
plt.plot(pd.date_range("00:00", "23:59", freq="30min"), X_scaled.T, color='k', alpha=0.02)
plt.title('Raw data')
plt.grid(True)
plt.show()

############################
# dimensionality reduction #
############################
p_components = 5
start = time.clock()
pca = decomposition.PCA(n_components=p_components, whiten=False)
pca.fit(X_scaled)
X_scores = pca.transform(X_scaled)
print('PCA done in %.4f seconds...' % (time.clock() - start))
print('Reduced dimensions: %s' % (str(X_scores.shape)))
print('Explained variance: %s' % (str(pca.explained_variance_ratio_)))

###########################
#   plot pca components   #
###########################
fig = plt.figure(1, facecolor='w')
for c in range(p_components):
    plt.plot(pca.components_[c, :], label='Component %d (var %.2f)' % ((c + 1), pca.explained_variance_ratio_[c]))
plt.title('Principal components')
plt.grid(True)
plt.legend()
plt.show()

################################
#     k-means centroids        #
# for non-uniform binary split #
################################
number_clusters = 5
# Centroids = clusterinit.GetCentroids().nubs(X_scaled, number_clusters)
# fig = plt.figure(1, facecolor='w')
# plt.plot(np.array(Centroids).T, linewidth=2)
# plt.show()

###########################
#   k-means clustering    #
###########################
start = time.clock()
number_clusters = 5
km = KMeans(number_clusters)
# km = KMeans(number_clusters, init=np.array(Centroids), n_init=1)
km.fit(X_scaled)
# km.fit(X_scores)
print('K-means done in %.4f seconds...' % (time.clock() - start))

# # reconstructed_centroids = pca.inverse_transform(km.cluster_centers_)
reconstructed_centroids = km.cluster_centers_

for i, centroid in enumerate(reconstructed_centroids):
    fig = plt.figure(2 + i, facecolor='w')
    indices = [j for j, x in enumerate(km.labels_.tolist()) if x == i]
    plt.plot(X_scaled[indices].T, color='k', alpha=0.02)
    plt.hold(True)
    plt.plot(centroid, label='Centroid %d' % (i), linewidth=3, color='#0099FF', alpha=0.9)
    plt.title('%s Centroid' % (season))
    plt.grid(True)
    plt.legend()
    plt.hold(False)
    # plt.savefig('profile_%d.png' % (i))
plt.show()

# # add cluster index to dataframe #
# df['Clusters'] = km.labels_

# ######################################
# # plot in browser with labels #
# ######################################
# # add cluster index to dataframe #
# # df['Clusters'] = km.labels_
# # figb, axb = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'), figsize=(12,10))
# # scatter = axb.scatter(X_scores[:, 0], X_scores[:, 1], c=km.labels_, alpha=0.8, cmap=plt.cm.jet)
# # axb.grid(color='white', linestyle='solid')
# # axb.set_title("Scatter", size=20)
# # tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=df['Labels'].values.tolist())
# # mpld3.plugins.connect(figb, tooltip)
# # mpld3.show()

# #############################################
# # 3d scatter plot for PCA with 3 components #
# #############################################
# # fig = plt.figure(7, facecolor='w')
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(X_scores[:, 0], X_scores[:, 1], X_scores[:, 2], c=km.labels_)
# # plt.title('3d PCA scores')
# # plt.show()

########################
# plot to google earth #
########################
# color_list = ['yellow', 'blue', 'red', 'orange', 'green']
icon_list = ['1.png', '2.png', '3.png', '4.png', '5.png']
kml = simplekml.Kml()
multipnt = kml.newmultigeometry(name="Clusters")

for i, item in enumerate(df['FeederName'].unique()):
    temp_dict = dict(collections.Counter(df.loc[df['FeederName'] == item]['Clusters'].values))
    clu = max(temp_dict.items(), key=operator.itemgetter(1))[0]
    Lon = feeder_dict[item][1]
    Lat = feeder_dict[item][2]
    if Lon != 'NULL' and Lat != 'NULL':
        # change GPS format - remove n/s/e/w and replace with signs #
        if 'e' in Lon.lower():
            Lon = '+' + str(Lon[:-1])
        if 'w' in Lon.lower():
            Lon = '-' + str(Lon[:-1])
        if 's' in Lat.lower():
            Lat = '-' + str(Lat[:-1])
        if 'n' in Lat.lower() and Lat != 'NULL':
            Lat = '+' + str(Lat[:-1])
        pnt = kml.newpoint(coords=[(Lon, Lat)])
        # pnt.style.iconstyle.color = kml_colors.color_dict[color_list[clu]]
        pnt.style.iconstyle.icon.href = icon_list[clu]
        pnt.style.iconstyle.scale = 0.9
        cluster = ''
        for k, v in temp_dict.items():
            cluster = cluster + ('Cluster %d: %.4f %% \n' % (k, float(v / len(df) * 100)))
        picpath = 'profile_%d.png' % (clu)
        pnt.description = 'Feeder Description: %s \n\n Cluster: %d \n\n %s \n\n %s' % (str(item), clu, cluster, '<img src="' + picpath + '" alt="picture" width="380" height="260" align="left" />')
kml.save("%s_clusters.kml" % (season))
os.system("%s_clusters.kml" % (season))
