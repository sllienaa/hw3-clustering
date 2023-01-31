import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations
        a score of -1 means the clustering is incorrect, 0 is indifferent clustering, 1 is clear distinct clusters

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        # initialize array to contain all the scores for observation
        scores = []

        # Go through each data point index
        for observation_index in range(0, X.shape[0]):
            # initialize list to hold intercluster distances
            a = []
            # initialize current cluster value
            current_cluster = y[observation_index]
            # calculate a
            # indices for points in the same cluster
            this_cluster = np.where(y == current_cluster)[0]
            # for other points within the same cluster, go through the indices
            for within_cluster_point in this_cluster:
                # make sure you don't include the distance between the current observation to itself
                if within_cluster_point != observation_index:
                    # add distance value to the running a list
                    a.append(cdist([np.array(X[observation_index])], [np.array(X[within_cluster_point])], 'euclidean'))
            # calculate the final value for a
            final_a = np.average(a)

            # calculate b
            # initialize list
            final_b = []
            # for each possible classification label
            for label in range(max(y) + 1):
                # get other points that are not in the same cluster as the current point
                if label == current_cluster:
                    continue
                else:
                    # initialize list to hold nearest cluster average distances
                    b = []
                    # for each other label, calculate distance between current point and other label, in a list b
                    for other_cluster_point in np.where(y == label)[0]:
                        b.append(cdist([np.array(X[observation_index])], [np.array(X[other_cluster_point])], 'euclidean'))
                # get the average distance value for each label
                final_b.append(np.average(b))
            # get the minimum of the average distance values for each label, this is the final b
            final_b = np.min(final_b)

            # calculate silhouette score
            # add score to the scores array
            scores.append((final_b - final_a) / max(final_a, final_b))

        return np.array(scores)
