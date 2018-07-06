import numpy as np
from common import *


__all__ = ['Distance',
           'Mahalanobis']


class Distance(object):
    """
    Parent class for all the distance type methods
    """

    def __init__(self,
                 samples=None,
                 names=None,
                 index=None):
        """
        Class constructor
        :param samples: List of sample dictionaries
        :param names: Name of the columns or dimensions
        :param index: Name of index column
        :return: _Distance object
        """
        self.samples = samples
        self.nsamp = len(samples)
        self.names = names
        self.index = index

        self.matrix = None
        self.center = None

    def __repr__(self):
        return "<Distance class object at {}>".format(hex(id(self)))

    def sample_matrix(self):
        """
        Method to convert sample dictionaries to sample matrix
        :return: Numpy matrix with columns as dimensions and rows as individual samples
        """
        # dimensions of the sample matrix
        nsamp = len(self.samples)
        nvar = len(self.names)

        # copy data to matrix
        samp_matrix = np.matrix([[self.samples[i][self.names[j]] for j in range(0, nvar)]
                                 for i in range(0, nsamp)])
        self.matrix = samp_matrix

    def cluster_center(self,
                       method='median'):
        """
        Method to determine cluster center of the sample matrix
        :param method: Type of reducer to use. Options: 'mean', 'median', 'percentile_x'
        :return: Cluster center (vector of column/dimension values)
        """
        if self.matrix is not None:
            if method == 'median':
                self.center = np.array(np.median(self.matrix, axis=0))[0]
            elif method == 'mean':
                self.center = np.array(np.mean(self.matrix, axis=0))[0]
            elif 'percentile' in method:
                perc = int(method.replace('percentile', '')[1:])
                self.center = np.array(np.percentile(self.matrix, perc, axis=0))[0]
            else:
                raise ValueError("Invalid or no reducer")
        else:
            raise ValueError("Sample matrix not found")


class Mahalanobis(Distance):
    """
    Class for calculating Mahalanobis distance from cluster center
    """

    def __init__(self,
                 samples=None,
                 names=None,
                 index=None):
        """
        Class constructor
        Class constructor
        :param samples: List of sample dictionaries
        :param names: Name of the columns or dimensions
        :param index: Name of index column
        :return: Mahalanobis object
        """

        super(Mahalanobis, self).__init__(samples,
                                          names,
                                          index)

        self.inverse = None
        self.distance = None

    def __repr__(self):
        return "<Mahalanobis class object at {}>".format(hex(id(self)))

    def covariance(self,
                   inverse=False):
        """
        Method to calculate a covariance matrix for a given sample matrix
        where rows are samples, columns are dimensions
        :param inverse: Should the inverse matrix be calculated
        :return: Covariance or inverse covariance matrix (numpy.matrix object)
        """
        cov_mat = np.cov(self.matrix,
                         rowvar=False)

        if inverse:
            # Inverse using SVD
            u, s, v = np.linalg.svd(cov_mat)

            try:
                return np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)

            except ValueError:
                return None
        else:
            return np.matrix(cov_mat)

    def difference(self,
                   transpose=False):
        """
        Method to calculate difference from scene center
        :return: matrix (numpy.ndarray)
        """
        center = self.center

        diff_matrix = np.matrix(np.apply_along_axis(lambda row: np.array(row) - center,
                                                    axis=1,
                                                    arr=np.array(self.matrix)))

        if transpose:
            return diff_matrix.T
        else:
            return diff_matrix

    def calc_distance(self):
        """
        Method to calculate mahalanobis distance from scene center
        :return: scalar value
        """
        inv_cov_matrix = self.covariance(True)

        diff = self.difference(False)
        transpose_diff = self.difference(True)

        mdist = np.zeros(self.nsamp)

        for i in range(0, self.nsamp):

            if inv_cov_matrix is None:
                mdist[i] = np.nan
                continue

            prod = np.array(
                            np.dot(
                                np.dot(diff[i, :],
                                       inv_cov_matrix),
                                transpose_diff[:, i])
                        )[0][0]

            if prod < 0:
                mdist[i] = np.nan
            else:
                mdist[i] = np.sqrt(prod)

        return list(mdist)
