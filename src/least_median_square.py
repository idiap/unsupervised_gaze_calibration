'''
This script consist in functions that compute the least median square estimation of the parametesr
of a given model on given data.

Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
Written by Remy Siegfried <remy.siegfried@idiap.ch>

This file is part of unsupervised_gaze_calibration.

unsupervised_gaze_calibration is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

unsupervised_gaze_calibration is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with unsupervised_gaze_calibration. If not, see <http://www.gnu.org/licenses/>.
'''
from copy import deepcopy
import numpy as np


def weighted_median(data, weights):
    """ from https://gist.github.com/tinybike/d9ff1dad515b66cc0d87 (author: Jack Peterson)
        <data> (list or numpy.array): data
        <weights> (list or numpy.array): weights
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median


def all_different(vec_list, tol=1.):
    if len(vec_list) == 1:
        return True

    for vec in vec_list:
        diff_list = np.sum((np.array(vec_list) - np.array(vec))**2, axis=1)
        n_similar = np.sum(map(int, diff_list < tol))
        if n_similar > 1:  # vec will be close to itself, so 1 similar vector is correct
            return False

    return True


def leastMedianOfSquare_regression(data_x, data_y, weights=None,
                                   computeParameters=None, computeResiduals=None,
                                   nb_iter=500, nb_samples=1, return_indexes=False):
    """ Performs a lest median of square regression using the RANSAC algorithm and return the best found parameters.
        <data_x> and <data_y> numpy.array of dimension (N, dim_x) and (N, dim_y)
        <computeParameters>: a function that takes as input a list of x and a list of y and return the estimated
            parameters matrix (see below for an example)
        <computeResidual>: a function that takes as input a x, a y and a parameter matrix and return the residual
            (see below for an example)
        <nb_iter> is the number of iteration in RANSAC
        <nb_samples> is the number of samples that are used to compute the regression proposal
        Note: default <computeParameters> and <computeResidual> correspond to a constant bias model, where x and y have
            the same dimension """

    def computeParameters_constantBias(x_list, y_list, regularization=False):
        """ Compute a constant bias that minimize e = y - (x - theta)
            <x_list> has the size (N, dim_x) and <y_list> has the size (N, dim_y) """
        theta = np.mean(x_list - y_list, 0)
        return theta

    def computeResiduals_constantBias(x_list, y_list, theta):
        """ Compute the residual as y - (x - theta)
            <x_list> has the size (N, dim_x) and <y_list> has the size (N, dim_y)
            Should return a numpy.array scalar of size (N, 1) """
        return np.reshape([np.linalg.norm(y - (x - theta)) for x, y in zip(x_list, y_list)], (-1, 1))

    # Use default functions if not given
    computeParameters = computeParameters_constantBias if computeParameters is None else computeParameters
    computeResiduals = computeResiduals_constantBias if computeResiduals is None else computeResiduals

    # RANSAC with Least Median Square (LMedS) to filter outliers
    theta_best, median_best = None, np.inf
    for i in range(0, nb_iter):
        # Select a few x and y and estimate regression parameters
        while True:
            indexes = range(len(data_x))
            np.random.shuffle(indexes)
            indexes = indexes[0:nb_samples]
            det = np.linalg.det(np.dot(data_x[indexes].transpose(), data_x[indexes]))
            if all_different(data_x[indexes], tol=0.01) and det != 0:
                break

        theta = computeParameters(data_x[indexes], data_y[indexes], regularization=False)

        # Compute the median of square residuals
        residuals = computeResiduals(data_x, data_y, theta)
        if weights is not None:
            QPercentile = weighted_median(np.array(residuals)**2, weights)
        else:
            QPercentile = np.percentile(np.array(residuals)**2, 50)

        # Update best parameters if needed
        if QPercentile < median_best:
            median_best = QPercentile
            theta_best = deepcopy(theta)

    # Filter data
    kept_indexes = []
    for i, (x, y) in enumerate(zip(data_x, data_y)):
        residuals = computeResiduals([x], [y], theta_best)
        if np.array(residuals)**2 <= median_best:
            kept_indexes.append(i)

    # Least Mean Square (LMS) on remaining points
    theta_final = computeParameters(data_x[kept_indexes], data_y[kept_indexes], regularization=True)

    if return_indexes:
        return theta_final, kept_indexes
    else:
        return theta_final

