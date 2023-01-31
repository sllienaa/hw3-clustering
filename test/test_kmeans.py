# Write your k-means unit tests here
import numpy as np
from scipy.spatial.distance import cdist
import cluster
import pytest
from sklearn.cluster import KMeans

def test_kmeans():
    """
    Unit tests for kmeans.
    """

    # Create data
    mat, labels = cluster.utils.make_clusters()

    # fit and predict labels using my implementation
    my_kmeans = cluster.KMeans(k=4)
    my_kmeans.fit(mat)
    my_kmeans.predict(mat)
    # make sure labels size matches the predictions size
    assert len(my_kmeans.predict(mat)) == len(labels), "Predcited labels are not the same dimensions as labels!" # check that they are the same length

