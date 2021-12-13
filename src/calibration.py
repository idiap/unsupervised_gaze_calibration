'''
This script consist in functions to compute calibration parameters from a list of samples using
different models (compute_calibration) and apply calibration parameters to gaze data
(calibrate_gaze)

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

import numpy as np

from src.least_median_square import leastMedianOfSquare_regression
from src.geometry import yawElevationToVector, vectorToYawElevation

DEFAULT_PARAMETERS = (np.diag([1., 1.]), np.zeros((2, 2)), np.zeros((2, 1)))


def compute_samples_calib_weights(samples):
    """ Compute a weight for each calibration sample to balance samples across targets.
        Thus, target with less samples wil lreceive a higher weight, following the formula:
        w(j) = 1 / (J * N(j)), with
            w(j) the weight of samples associated to the target j,
            J the number of targets, and
            N(j) the number of calibration sample labelled as 'looking to target j' """

    if 'target_name' not in samples.keys():
        print('[calibration::compute_samples_calib_weights] no "target_name" key in samples')
        return np.array([])
    
    # Each target has the same weight, which is distributed among the samples corresponding to it
    targets, counts = np.unique(samples['target_name'], return_counts=True)
    weights = 1. / len(targets) / counts
    weight_dict = dict(zip(targets, weights))
    weight_list = [weight_dict[target_name[0]] for target_name in samples['target_name']]

    # Sanity check
    if not np.allclose(np.sum(weight_list), 1):
        raise Exception('Error in weight allocation: sum is not 1')
    
    return np.array(weight_list)


def compute_calibration_constant(samples, weights=None, ransac_n_iter=500):
    """ Model: y = (gaze) + C """

    if len(samples['gaze']) < 2:
        print('[calibration::compute_calibration_constant] not enough calibration samples')
        return DEFAULT_PARAMETERS
    
    def computeParameters(x_list, y_list, regularization=False):
        return np.mean(y_list - x_list, axis=0)

    def computeResiduals(x_list, y_list, bias):
        return y_list - (x_list + bias)

    # Yaw regression
    theta_yaw = leastMedianOfSquare_regression(samples['gaze'][:, [0]], samples['target'][:, [0]],
                                               weights, computeParameters, computeResiduals,
                                               nb_iter=ransac_n_iter, nb_samples=1)

    # Elevation regression
    theta_ele = leastMedianOfSquare_regression(samples['gaze'][:, [1]], samples['target'][:, [1]],
                                               weights, computeParameters, computeResiduals,
                                               nb_iter=ransac_n_iter, nb_samples=1)

    # Standardize parameters
    A, B, C = DEFAULT_PARAMETERS
    C = np.array([[theta_yaw[0]], [theta_ele[0]]], dtype=np.float32)
    return A, B, C


def compute_calibration_linearGaze(samples, weights=None, ransac_n_iter=500,
                                   linearRidge_mu0=[1., 0., 0.],
                                   linearRidge_Lambda=[1e4, 1e4, 0.]):
    """ Model: y = A * (gaze) + C """

    if len(samples['gaze']) < 4:
        print('[calibration::compute_calibration_constant] not enough calibration samples')
        return DEFAULT_PARAMETERS
    
    mu0 = np.reshape(linearRidge_mu0, (-1, 1))
    Lambda = np.diag(linearRidge_Lambda)
    
    def computeParameters(x_list, y_list, regularization=False):
        """ without regularization: Y = XA    => A = X^-1 * Y
            with regularization:    A = (X^T * X + Lambda)^-1 * (X^T * Y + Lambda * mu0) """
        if regularization:
            theta = np.linalg.solve(a=(np.dot(x_list.transpose(), x_list) + Lambda),
                                    b=(np.dot(Lambda, mu0) + np.dot(x_list.transpose(), y_list)))
        else:
            theta = np.linalg.solve(a=(np.dot(x_list.transpose(), x_list)),
                                    b=(np.dot(x_list.transpose(), y_list)))  # ax=b
        return theta

    def computeResiduals(x_list, y_list, theta):
        return y_list - np.dot(x_list, theta)

    # Yaw regression
    x = np.hstack([samples['gaze'][:, [0]], samples['gaze'][:, [1]], np.ones((samples['gaze'].shape[0], 1))])
    y = samples['target'][:, [0]]
    theta_yaw = leastMedianOfSquare_regression(x, y,
                                               weights, computeParameters, computeResiduals,
                                               nb_iter=ransac_n_iter,
                                               nb_samples=3).flatten()

    # Elevation regression
    x = np.hstack([samples['gaze'][:, [1]], samples['gaze'][:, [0]], np.ones((samples['gaze'].shape[0], 1))])
    y = samples['target'][:, [1]]
    theta_ele = leastMedianOfSquare_regression(x, y,
                                               weights, computeParameters, computeResiduals,
                                               nb_iter=ransac_n_iter,
                                               nb_samples=3).flatten()
    # Reformat parameters
    A = np.array([[theta_yaw[0], theta_yaw[1]],
                  [theta_ele[1], theta_ele[0]]], dtype=np.float32)
    B = np.zeros((2, 2))
    C = np.array([[theta_yaw[2]], [theta_ele[2]]], dtype=np.float32)
    return (A, B, C)


def compute_calibration_linearHeadGaze(samples, weights=None, ransac_n_iter=500,
                                       linearRidge_mu0=[1., 0., 0., 0., 0.],
                                       linearRidge_Lambda=[1e4, 1e4, 1e4, 1e4, 0.]):
    """ Model: y = A * (gaze) + B * (headpose) C """
    
    if len(samples['gaze']) < 6:
        print('[calibration::compute_calibration_constant] not enough calibration samples')
        return DEFAULT_PARAMETERS
    
    mu0 = np.reshape(linearRidge_mu0, (-1, 1))
    Lambda = np.diag(linearRidge_Lambda)
    
    def computeParameters(x_list, y_list, regularization=False):
        """ without regularization: Y = XA    => A = (X^T * X)^-1 * (X^T * Y)
            with regularization:    A = (X^T * X + Lambda)^-1 * (X^T * Y + Lambda * mu0) """
        if regularization:
            theta = np.linalg.solve(a=(np.dot(x_list.transpose(), x_list) + Lambda),
                                    b=(np.dot(Lambda, mu0) + np.dot(x_list.transpose(), y_list)))
        else:
            theta = np.linalg.solve(a=(np.dot(x_list.transpose(), x_list)),
                                    b=(np.dot(x_list.transpose(), y_list)))  # ax=b
        return theta

    def computeResiduals(x_list, y_list, theta):
        return y_list - np.dot(x_list, theta)

    # Yaw regression
    x = np.hstack([samples['gaze'][:, [0]], samples['gaze'][:, [1]],
                   samples['headpose'][:, [0]], samples['headpose'][:, [1]],
                   np.ones((samples['gaze'].shape[0], 1))])
    y = samples['target'][:, [0]]
    theta_yaw = leastMedianOfSquare_regression(x, y,
                                               weights, computeParameters, computeResiduals,
                                               nb_iter=ransac_n_iter,
                                               nb_samples=6).flatten()

    # Elevation regression
    x = np.hstack([samples['gaze'][:, [1]], samples['gaze'][:, [0]],
                   samples['headpose'][:, [1]], samples['headpose'][:, [0]],
                   np.ones((samples['gaze'].shape[0], 1))])
    y = samples['target'][:, [1]]
    theta_ele = leastMedianOfSquare_regression(x, y,
                                               weights, computeParameters, computeResiduals,
                                               nb_iter=ransac_n_iter,
                                               nb_samples=6).flatten()

    # Reformat parameters
    A = np.array([[theta_yaw[0], theta_yaw[1]],
                  [theta_ele[1], theta_ele[0]]], dtype=np.float32)
    B = np.array([[theta_yaw[2], theta_yaw[3]],
                  [theta_ele[3], theta_ele[2]]], dtype=np.float32)
    C = np.array([[theta_yaw[4]], [theta_ele[4]]], dtype=np.float32)
    return (A, B, C)


def compute_calibration_knn(samples, weights=None, ransac_n_iter=500, knn_n=10, knn_k=3):
    """ Model: y = (gaze) + C
        C computed through KNN (<knn_n> samples in memory, use <knn_k> nearest neighbours"""
    
    if len(samples['gaze']) < knn_n:
        print('[calibration::compute_calibration_constant] not enough calibration samples')
        return DEFAULT_PARAMETERS

    def computeParameters(x_list, y_list, regularization=False):
        """ Select <knn_n> random samples as reference points """
        indexes = range(len(x_list))
        np.random.shuffle(indexes)
        indexes = indexes[0:knn_n]
        theta = np.hstack([np.array(x_list)[indexes], np.array(x_list)[indexes] - np.array(y_list)[indexes]])
        return theta

    def computeResiduals(x_list, y_list, theta):
        """ Model: y = x + b"""
        dx = len(x_list[0])
        ref_list = theta[:, 0:dx]
        bias_list = theta[:, dx:]
        residuals = []
        for x, y in zip(x_list, y_list):
            distance = np.array([np.linalg.norm(x - ref) for ref in ref_list])
            threshold = distance[np.argsort(distance)][knn_k-1]
            indexes = np.where(distance <= threshold)
            bias = np.mean(bias_list[indexes])
            residuals.append(np.linalg.norm(y - (x - bias)))
        return np.reshape(residuals, (-1, 1))

    x = np.hstack([samples['gaze'][:, [0]], samples['gaze'][:, [1]]])
    y = np.hstack([samples['target'][:, [0]], samples['target'][:, [1]]])
    parameters = leastMedianOfSquare_regression(x, y,
                                                weights, computeParameters, computeResiduals,
                                                nb_iter=ransac_n_iter, nb_samples=10)
    return parameters
    

def compute_calibration(samples_calib, calib_model='constant', weight_calib_samples=True, ransac_n_iter=500,
                        linearRidge_mu0=[1., 0., 0., 0., 0.], linearRidge_Lambda=[1e4, 1e4, 1e4, 1e4, 0],
                        knn_n=10, knn_k=3):

    if 'gaze' not in samples_calib.keys() or len(samples_calib['gaze']) == 0:
        print('[calibration::compute_calibration] no calibration samples')
        return DEFAULT_PARAMETERS
    
    # Compute weights for each calibration sample
    weights = compute_samples_calib_weights(samples_calib) if weight_calib_samples else None

    linearRidge_mu0 = np.array(linearRidge_mu0)
    linearRidge_Lambda = np.array(linearRidge_Lambda)
    
    # Model y = A*(gaze) + B*(headpose) + C
    # Thus, parameters=(A,B,C)

    if calib_model == 'none':
        parameters = (np.diag([1., 1.]), np.zeros((2, 2)), np.zeros((2, 1)))
    elif calib_model == 'constant':
        parameters = compute_calibration_constant(samples_calib, weights, ransac_n_iter)
    elif calib_model == 'linearGaze':
        parameters = compute_calibration_linearGaze(samples_calib, weights, ransac_n_iter,
                                                    linearRidge_mu0[[0, 1, 4]], linearRidge_Lambda[[0, 1, 4]])
    elif calib_model == 'linearHeadGaze':
        parameters = compute_calibration_linearHeadGaze(samples_calib, weights, ransac_n_iter,
                                                        linearRidge_mu0, linearRidge_Lambda)
    elif calib_model == 'knn':
        parameters = compute_calibration_knn(samples_calib, weights, ransac_n_iter,
                                             knn_n, knn_k)
    else:
        raise Exception('Unknown calibration model: {}'.format(calib_model))

    return parameters


def calibrate_gaze(gaze, headpose=None, calib_parameters=None, calib_model='constant', already_2D=False,
                   knn_k=3):
    """ Apply calibration to input gaze and headpose angles with the following model:
        y = A * (gaze) + B * (headpose) + C
        <gaze>: (N, 3) list of 3D gaze vectors (x, y, z)
        <headpose>: (N, 3) list of headpose angles (roll, pitch, yaw)
        <calib_parameters>: list of three parameters A (2x2 matrix), B (2x2 matrix), and C (2x1 matrix)"""

    if calib_parameters is None:
        return gaze
    
    # Convert gaze and headpose to 2D angles
    if not already_2D:
        gaze = np.array([vectorToYawElevation(g) for g in gaze])
        if headpose is not None:
            headpose = np.array([[hp[2], -hp[1]]  for hp in headpose]) * 180/np.pi

    # Compute calibrated gaze
    if calib_model in ['none', 'constant', 'linearGaze', 'linearHeadGaze']:
        A, B, C = calib_parameters
        # Use transposed as data are given in rows (i.e. (N, 2) arrays)
        if headpose is not None:
            gaze_calib = np.dot(gaze, A.transpose()) + np.dot(headpose, B.transpose()) + C.transpose()
        else:
            gaze_calib = np.dot(gaze, A.transpose()) + C.transpose()
            
    elif calib_model == 'knn':
        ref_list = calib_parameters[:,0:2]
        bias_list = calib_parameters[:,2:]
        gaze_calib = np.array([]).reshape([0, 2])
        for g in gaze:
            distance = np.array([np.linalg.norm(g - ref) for ref in ref_list])
            threshold = distance[np.argsort(distance)][knn_k-1]
            indexes = np.where(distance <= threshold)
            bias = np.mean(bias_list[indexes], axis=0)
            gaze_calib = np.vstack([gaze_calib, (g - bias).reshape((-1, 2))])
            
    else:
        raise Exception('Unknown calibration model: {}'.format(calib_model))

    # Back to gaze input space
    if not already_2D:
        gaze_calib = np.array([yawElevationToVector(g).flatten() for g in gaze_calib])

    return gaze_calib
