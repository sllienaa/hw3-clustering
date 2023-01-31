import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        # attribute for k
        self.k = int(k)
        self.max_iter = int(max_iter)
        self.tol = tol


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        # error handling
        if self.k == 0:
            raise Exception("You cannot have a K of 0!")
        if self.k > len(mat):
            raise Exception("You can't have a higher K than the number of observations!")

        # get k random observations from the matrix as our starting centroids
        self.centroids = mat[np.random.choice(mat.shape[0], self.k, replace = True)]

        # calculate distance between each centroid and the data points
        # keep track of the number of iterations, making sure it is under the specified maximum iterations
        # initialize iterations
        iterations = 0
        # initialize new_error and error change to a very large value
        new_error = 10000
        delta_loss = 1000

        # while iterations is less than max iterations and error is greater than the tolerance
        while iterations < self.max_iter and delta_loss > self.tol:
            # assign each data point to a centroid based on the minimum distance to the centroids
            self.clusters = self._assign_clusters(mat)
            # get average of data points for each classification to make new centroids
            self.centroids = self.get_centroids(mat)
            # calculate the error of the new model
            error = self._get_error()  # get the mse of all points to each centroid

            # time to evaluate if error of our new model increased
            # calculate change in loss, getting absolute value
            delta_loss = np.abs(new_error - error)
            # reassign error to reflect new iteration
            new_error = error

            # add 1 iteration to keep count
            iterations += 1




    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        return self.clusters




    def _get_error(self) -> float:
        """
        Returns the error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        Private method.

        outputs:
            float
                the squared-mean error of the fit model
        """
        sse = 0
        for point in range(self._distances.shape[0]):
            sse = sse + np.square(self._distances[point][self.clusters[point]])

        return sse/self._distances.shape[0]


    def get_centroids(self, mat: np.ndarray) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        # initialize new matrix to store cluster centroids using old centroids array
        # public method
        centroids = np.empty(self.centroids.shape)
        # for each label
        for i in range(self.k):
            centroids[i] = np.mean(mat[self.clusters == i], axis=0)

        return centroids



    def _assign_clusters(self, mat: np.ndarray):
        # calculate euclidean distance from each centroid and data point (stored in np array)
        # stored in private variable because there's no need to make it accessible by the user
        # decided to hard code euclidean since k-means uses euclidean distance
        self._distances = cdist(mat, self.centroids, 'euclidean')
        # based on the distances, assign each point to a centroid using the minimum distance from the distances array by row
        clusters = []
        for i in range(self._distances.shape[0]):
            centroid_idx = np.argmin(self._distances[i])
            clusters.append(centroid_idx)

        #clusters = np.array([np.argmin(i) for i in self._distances])

        return np.array(clusters)
