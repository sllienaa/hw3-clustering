from cluster import Silhouette, make_clusters
import pytest
import numpy as np
from sklearn.metrics import silhouette_samples

def test_silhouette():
    """
    testing my silhouette score implementation
    """
    # initialize data
    clusters, labels = make_clusters(k=5, scale=0.3)
    # get silhouette score using my implementation
    my_scores = Silhouette().score(clusters, labels)
    # get scores using sklearn
    sklearn_scores = silhouette_samples(clusters, labels)
    # check that they are the same using np.allclose
    assert np.allclose(my_scores, sklearn_scores) == True

