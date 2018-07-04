import numpy as np
from common import *


class _Distance(object):
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
                return np.median(self.matrix, axis=0)
            elif method == 'mean':
                return np.mean(self.matrix, axis=0)
            elif 'percentile' in method:
                perc = int(method.replace('percentile', '')[1:])
                return np.percentile(self.matrix, perc, axis=0)
            else:
                raise ValueError("Invalid or no reducer")
        else:
            raise ValueError("Sample matrix not found")


class Mahalanobis(_Distance):
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
            return np.linalg.inv(cov_mat)
        else:
            return cov_mat

    def difference(self,
                   transpose=False):
        """
        Method to calculate difference from scene center
        :return: matrix (numpy.ndarray)
        """
        center = self.center

        diff_matrix = np.apply_along_axis(lambda row: np.array(row) - center,
                                          axis=0,
                                          arr=self.matrix)
        if transpose:
            return np.transpose(diff_matrix)
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

        mdist = list()

        for i in range(0, self.nsamp):
            mdist.append(np.sqrt(np.dot(np.dot(transpose_diff,
                                        inv_cov_matrix),
                                 diff)))

        return mdist
