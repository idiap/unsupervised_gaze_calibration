'''
This script consist in function that allow to select calibration samples in a list of sample given
some constraints.

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

from src.data_loader import stack_in_dict
from src.geometry import vectorToYawElevation


def check_physical_constraints(feature_2D, target_2D, constraints=[30., 30., 30., 30.]):
    """ Check if physical constraints are fulfilled (return True) or not (return False) between a given feature
            and the target position, both expressed in 2D angles (unit is degree).
        <feature_2D> (list/array, len=2) 2D angular position of a feature to compare with target position,
            typically head pose or gaze direction
        <targe_2D> (list/array, len=2) 2D angular position of the target
        <constraints> (list/array, len=4) maximal tolerated difference for each direction
            (yaw+, yaw-, elevation +, elevation-) """
    
    constraint_yaw = True
    if feature_2D[0] < target_2D[0] and abs(feature_2D[0] - target_2D[0]) > constraints[0]:  # target to the right
        constraint_yaw = False
    elif feature_2D[0] > target_2D[0] and abs(feature_2D[0] - target_2D[0]) > constraints[1]:  # target to the left
        constraint_yaw = False

    constraint_elevation = True
    if feature_2D[1] < target_2D[1] and abs(feature_2D[1] - target_2D[1]) > constraints[2]:  # target up
        constraint_elevation = False
    elif feature_2D[1] > target_2D[1] and abs(feature_2D[1] - target_2D[1]) > constraints[3]:  # target down
        constraint_elevation = False

    return constraint_yaw and constraint_elevation


def get_samples_calib_oracle(data):
    """ Keep samples for which: VFOA ground truth is a target with known target position
        <data>: (dict) dictonary with features as keys and (N, l) sample lists as values (see src/data_loader.py) """
    
    samples_calib = {}
    for i, gaze in enumerate(data['subject']['gaze']):
        if np.isnan(gaze).any():
            continue
        target_name = data['subject']['vfoa_gt'][i, 0]
        if target_name in data['targets'].keys():
            target_pos = data['targets'][target_name][i, 0:3]
            if not np.isnan(target_pos).any():
                # Transform to 2D angles
                gaze_2D = vectorToYawElevation(gaze)
                target_2D = vectorToYawElevation(target_pos - data['subject']['eye'][i])
                headpose_deg = np.array(data['subject']['headpose'][i]) * 180/np.pi
                headpose_CCS_2D = [headpose_deg[2], -headpose_deg[1]]

                stack_in_dict(samples_calib, 'identifier', [data['subject']['identifier'][i]])
                stack_in_dict(samples_calib, 'frameIndex', [data['subject']['frameIndex'][i]])
                stack_in_dict(samples_calib, 'vfoa_gt', [data['subject']['vfoa_gt'][i]])
                stack_in_dict(samples_calib, 'gaze', np.reshape(gaze_2D, (-1, 2)))
                stack_in_dict(samples_calib, 'headpose', np.reshape(headpose_CCS_2D, (-1, 2)))
                stack_in_dict(samples_calib, 'target', np.reshape(target_2D, (-1, 2)))
                stack_in_dict(samples_calib, 'target_name', np.reshape([target_name], (-1, 1)))

    return samples_calib

    
def get_samples_calib_speaking(data, thresh_physical_head=[30., 30., 30., 30.],
                               thresh_physical_gaze=[90., 90., 90., 90.], filter_no_annot=False):
    """ Keep samples for which: a single target is speaking and physical constraints are fulfilled
        <data>: (dict) dictonary with features as keys and (N, l) sample lists as values (see src/data_loader.py)
        <thresh_physical_head>: (list, len=4), threshold for physical constraint prior applied on head pose
            (yaw+, yaw-, elevation+, elevation-)
        <thresh_physical_gaze>: (list, len=4) same applied on gaze
        <filter_no_annot>: (bool) if True, remove samples that were not annotated (useful to filter out blinks)"""
    
    samples_calib = {}
    for i, gaze in enumerate(data['subject']['gaze']):
        if np.isnan(gaze).any():
            continue
        if filter_no_annot and data['subject']['vfoa_gt'][i, 0] in ['na', 'no_data', 'blink', 'blinking']:
            continue

        # Use conversation prior to get the VFOA target (i.e. speaker if there is only one)
        speakers = []
        if data['subject']['speaking'][i, 0] == 1:
            speakers.append('subject')
        for target_name in data['targets'].keys():
            if data['targets'][target_name][i, 3] == 1:
                speakers.append(target_name)
        target_name = speakers[0] if len(speakers) == 1 else None

        if target_name in data['targets'].keys():
            target_pos = data['targets'][target_name][i, 0:3]
            if not np.isnan(target_pos).any():
                gaze_2D = vectorToYawElevation(gaze)
                target_2D = vectorToYawElevation(target_pos - data['subject']['eye'][i])
                
                # Headpose is (0,0) as we work in Head Coordinate System.
                # Still store real headpose, as it can be useful in calibration model
                headpose_CCS_deg = np.array(data['subject']['headpose'][i]) * 180/np.pi
                headpose_CCS_2D = [headpose_CCS_deg[2], -headpose_CCS_deg[1]]
                headpose_HCS = [0., 0.]
                
                # Check physical constraints
                if check_physical_constraints(headpose_HCS, target_2D, thresh_physical_head) and \
                   check_physical_constraints(gaze_2D, target_2D, thresh_physical_gaze):
                    stack_in_dict(samples_calib, 'identifier', [data['subject']['identifier'][i]])
                    stack_in_dict(samples_calib, 'frameIndex', [data['subject']['frameIndex'][i]])
                    stack_in_dict(samples_calib, 'vfoa_gt', [data['subject']['vfoa_gt'][i]])
                    stack_in_dict(samples_calib, 'gaze', np.reshape(gaze_2D, (-1, 2)))
                    stack_in_dict(samples_calib, 'headpose', np.reshape(headpose_CCS_2D, (-1, 2)))
                    stack_in_dict(samples_calib, 'target', np.reshape(target_2D, (-1, 2)))
                    stack_in_dict(samples_calib, 'target_name', np.reshape([target_name], (-1, 1)))
    return samples_calib


def get_samples_calib_manip(data, w_interval=[-30, -20], thresh_physical_head=[30., 30., 30., 30.],
                               thresh_physical_gaze=[90., 90., 90., 90.], filter_no_annot=False):
    """ Keep samples for which: a grasping/releasing action occurs in the future and
            physical constraints are fulfilled
        <data>: (dict) dictonary with features as keys and (N, l) sample lists as values (see src/data_loader.py)
        <w_interval> (list, len=2) interval during which the subject looks at the action location 
            relatively to the action time
        <thresh_physical_head>: (list, len=4), threshold for physical constraint prior applied on head pose
            (yaw+, yaw-, elevation+, elevation-)
        <thresh_physical_gaze>: (list, len=4) same applied on gaze
        <filter_no_annot>: (bool) if True, remove samples that were not annotated (useful to filter out blinks)"""

    samples_calib = {}
    for i, action in enumerate(data['subject']['action']):
        # Look for grasp/releas action
        action_name, action_location = action[0].split(':')
        if action_name not in ['grasp', 'release']:
            continue

        for j in range(w_interval[0], w_interval[1]):
            if np.isnan(data['subject']['gaze'][i+j]).any():
                continue
            if filter_no_annot and data['subject']['vfoa_gt'][i+j, 0] in ['na', 'no_data', 'blink', 'blinking']:
                continue

            # Use manipulation prior to get the VFOA target (i.e. location of the future grasp/release)
            target_name = action_location

            if target_name in data['targets'].keys():
                target_pos = data['targets'][target_name][i+j, 0:3]
                if not np.isnan(target_pos).any():
                    gaze_2D = vectorToYawElevation(data['subject']['gaze'][i+j])
                    target_2D = vectorToYawElevation(target_pos - data['subject']['eye'][i+j])

                    # Headpose is (0,0) as we work in Head Coordinate System.
                    # Still store real headpose, as it can be useful in calibration model
                    headpose_CCS_deg = np.array(data['subject']['headpose'][i]) * 180/np.pi
                    headpose_CCS_2D = [headpose_CCS_deg[2], -headpose_CCS_deg[1]]
                    headpose_HCS = [0., 0.]

                    # Check physical constraints
                    if check_physical_constraints(headpose_HCS, target_2D, thresh_physical_head) and \
                       check_physical_constraints(gaze_2D, target_2D, thresh_physical_gaze):
                        stack_in_dict(samples_calib, 'identifier', [data['subject']['identifier'][i+j]])
                        stack_in_dict(samples_calib, 'frameIndex', [data['subject']['frameIndex'][i+j]])
                        stack_in_dict(samples_calib, 'vfoa_gt', [data['subject']['vfoa_gt'][i+j]])
                        stack_in_dict(samples_calib, 'gaze', np.reshape(gaze_2D, (-1, 2)))
                        stack_in_dict(samples_calib, 'headpose', np.reshape(headpose_CCS_2D, (-1, 2)))
                        stack_in_dict(samples_calib, 'target', np.reshape(target_2D, (-1, 2)))
                        stack_in_dict(samples_calib, 'target_name', np.reshape([target_name], (-1, 1)))
    return samples_calib


def select_samples(samples, n_samples='all', selection='random'):
    """ Keep only <n_samples> in <samples> using the <selection> method
        <samples>: (dict) dictonary with features as keys and (N, l) sample lists as values (see src/data_loader.py) """
    
    n_samples_current = len(samples['identifier'])
    if n_samples == 'all' or n_samples_current <= n_samples:
        return samples
    else:
        if selection == 'firsts':
            mask = range(n_samples)
        elif selection == 'random':
            mask = range(n_samples_current)
            np.random.shuffle(mask)
            mask = mask[:n_samples]
        else:
            raise Exception('Unknown selection method: {}'.format(selection))

        samples_downsampled = {}
        for key, val in samples.items():
            samples_downsampled[key] = val[mask]
        return samples_downsampled
        

def get_samples_calib(data_calib, prior='oracle', n_samples='all', selection='random',
                      w_interval=[-30, -20],
                      thresh_physical_head=[30., 30., 30., 30.], thresh_physical_gaze=[90., 90., 90., 90.]):
    """ <data_calib>: (dict) dictonary with keys 'subject' and 'targets' (see src/data_loader.py) """

    # Gather all samples that fullfilled the prior
    if prior == 'oracle':
        samples_calib = get_samples_calib_oracle(data_calib)
    elif prior == 'speaking':
        samples_calib = get_samples_calib_speaking(data_calib, thresh_physical_head, thresh_physical_gaze)
    elif prior == 'manipulation':
        samples_calib = get_samples_calib_manip(data_calib, w_interval, thresh_physical_head, thresh_physical_gaze)
    else:
        raise Exception('Unknown calibration prior: {}'.format(prior))

    # Get <n_samples>
    if len(samples_calib.keys()) > 0:
        samples_calib = select_samples(samples_calib, n_samples, selection)

    return samples_calib
