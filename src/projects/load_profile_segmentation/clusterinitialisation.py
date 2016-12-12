# Projects: cluster_initialisation
#
# :Author: Jaco du Toit <jacowp357@gmail.com>
# :Date: 23/09/2015
# :Description: Fast cluster initialisation using suboptimal centroids
#               discovered by uniform binary split algorithm.
#
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import time


class GetCentroids():
    """
    This class determines cluster centroids for the k-means algorithm, and can
    be used to speed up the k-means algorithm.
    """
    def nubs(self, dataset, num_clusters):
        """
        This function uses non-uniform binary split to determine the initial centroids.

        - **parameters**, **types**, **return** and **return types**::
            :param arg1: dataset
            :param arg2: num_clusters
            :type arg1: numpy data matrix
            :type arg2: int
            :return: returns an array/matrix of centroids
            :rtype: numpy array/matrix of floats

        .. todo:: add a case where average distances are equal.
        """
        start = time.clock()
        km = KMeans(2)
        km.fit(dataset)
        print('K-means done in %.4f seconds, on data shape %s, clusters = %d' % (time.clock() - start, str(dataset.shape), 2))
        Centroids = []
        X_ubs = dataset
        for i in range(num_clusters):
            dist_from_cent_0 = euclidean_distances(km.cluster_centers_[0], X_ubs[km.labels_ == 0])
            dist_from_cent_1 = euclidean_distances(km.cluster_centers_[1], X_ubs[km.labels_ == 1])
            if dist_from_cent_0.sum() > dist_from_cent_1.sum():
                # choose centroid 1 and new data set is cluster 0 #
                X_ubs = X_ubs[km.labels_ == 0]
                Centroids.append(km.cluster_centers_[1])
            if dist_from_cent_1.sum() > dist_from_cent_0.sum():
                # choose centroid 0 and new data set is cluster 1 #
                X_ubs = X_ubs[km.labels_ == 1]
                Centroids.append(km.cluster_centers_[0])
            # TODO: case where average distance is the same #
            start = time.clock()
            km = KMeans(2)
            km.fit(X_ubs)
            print('K-means done in %.4f seconds, on data shape %s, clusters = %d' % (time.clock() - start, str(X_ubs.shape), 2))
        return Centroids
