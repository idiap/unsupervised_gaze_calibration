'''
This script consist in evaluation function that output different metrics like mean angular error
and vfoa accuracy.

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

import os
import numpy as np
import sklearn.metrics as sk_metrics

from src.geometry import yawElevationToVector, vectorToYawElevation, angleBetweenVectors_deg


def compute_mean_angular_error(gaze_list, eye_list, target_dict, vfoa_gt_list):
    """ Compute average angular error between the gaze and the ground truth VFOA target direction
        angular error 3D is the cosine distance between gaze and target's direction
        angular error 2D is the relative yaw and elevation error between gaze and target's direction (2 values)
        Return the number of samples used in evaluation, the average of both metrics and their standard deviation
        <gaze_list>: (N, 3) list of 3D gaze vectors (x, y, z)
        <eye_list>: (N, 3) list of 3D eye position (x, y, z)
        <target_dict>: (dict) dictionary of target data (3D position + speaking status)
            {'target_name': (N, 4) array}
        <vfoa_gt_list>: (N, 1) list of VFOA ground truth (i.e. target name or 'aversion') """
    
    angular_error_3D_list, angular_error_2D_list = [], []
    for i, gaze in enumerate(gaze_list):
        vfoa_gt = vfoa_gt_list[i, 0]
        eye = eye_list[i]
        target_gt = target_dict[vfoa_gt][i][0:3] if vfoa_gt in target_dict.keys() else np.nan * np.empty(3)
        if np.isnan(gaze).any() or np.isnan(eye).any() or np.isnan(target_gt).any():
            # Missing gaze, eye, or target data
            continue

        gaze_gt = target_gt - eye
        angular_error_3D_list.append(angleBetweenVectors_deg(gaze, gaze_gt))
        angular_error_2D_list.append(np.array(vectorToYawElevation(gaze)) - np.array(vectorToYawElevation(gaze_gt)))

    nb_eval_samples = len(angular_error_3D_list)

    return nb_eval_samples, np.nanmean(angular_error_3D_list), np.nanmean(angular_error_2D_list, axis=0), \
           np.std(angular_error_3D_list), np.std(angular_error_2D_list, axis=0)


def compute_vfoa_accuracy(vfoa_list, vfoa_gt_list, target_names):
    """ Compute VFOA accuracy and confusion matrix given a predicted and a ground truth VFOA list.
        Samples for which the ground is None, 'na', or not in the <target_names> list are ignored.
        <vfoa_list>: (N, 1) list of VFOA predictions
        <vfoa_gt_list>: (N, 1) list of ground truth VFOA
        <target_names>: (list) list of target names (len=number of different targets) """
    target_names = ['aversion'] + list(target_names)
    pred_list, gt_list = [], []
    for i, vfoa_gt in enumerate(vfoa_gt_list):
        if vfoa_gt not in target_names or i > len(vfoa_list) or vfoa_list[i] in [None, 'na']:
            continue
        
        pred_list.append(vfoa_list[i])
        gt_list.append(vfoa_gt)

    if len(gt_list) > 0:
        n_eval_samples = len(gt_list)
        accuracy = sk_metrics.accuracy_score(gt_list, pred_list)
        confusion_matrix = sk_metrics.confusion_matrix(gt_list, pred_list, labels=target_names)
        return n_eval_samples, accuracy, confusion_matrix
    else:
        return 0., np.nan, None
