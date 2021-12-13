'''
This script consist in function to estimate the VFOA.

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

from src.geometry import angleBetweenVectors_deg


def compute_vfoa_sample(eye, gaze, target_dict, threshold_vfoa=10.):
    """ Compute VFOA for one sample, given a set of targets
        <eye>: (len=3) 3D eye position (x, y, z)
        <gaze>: (len=3) 3D gaze vectors (x, y, z)
        <target_dict>: (dict) dictionary of target data (3D position + speaking status)
            {'target_name': (len=4) array}
        <threshold_vfoa>: (float) maximal tolerated cosine distance for a gaze direction to be allocatd to a target """
    if np.isnan(gaze).any():
        return 'na'
        
    nearest_target, nearest_distance = 'aversion', np.inf
    n_target = 0
    for target_name, target_data in target_dict.items():
        if not np.isnan(target_data[0:3]).any():
            n_target += 1
            dist = angleBetweenVectors_deg(gaze, target_dict[target_name][0:3] - eye)
            if dist < threshold_vfoa and dist < nearest_distance:
                nearest_target, nearest_distance = target_name, dist
                
    if n_target > 0 :
        return nearest_target
    else:
        return 'na'

    
def compute_vfoa(eye_list, gaze_list, target_dict, threshold_vfoa=10.):
    """ Compute VFOA for each gaze in <gaze_list>, given a set of targets
        <eye_list>: (N, 3) list of 3D eye position (x, y, z)
        <gaze_list>: (N, 3) list of 3D gaze vectors (x, y, z)
        <target_dict>: (dict) dictionary of target data (3D position + speaking status)
            {'target_name': (N, 4) array}
        <threshold_vfoa>: (float) maximal tolerated cosine distance for a gaze direction to be allocatd to a target """
    vfoa = []
    for i, gaze in enumerate(gaze_list):
        target_dict_sample = {}
        for target_name, target_data in target_dict.items():
            target_dict_sample[target_name] = target_data[i]
            
        vfoa.append(compute_vfoa_sample(eye_list[i], gaze, target_dict_sample, threshold_vfoa))

    return np.reshape(vfoa, (-1, 1))
