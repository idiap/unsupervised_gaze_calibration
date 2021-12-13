'''
This script consist in function to load/save data and configurations.

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
import json
import numpy as np


DEFAULT_CONFIG = {
    "calib_prior": "speaking",  # in ["oracle", "speaking", "manipulation"]
    "calib_n_samples": "all",  # number of samples use for calibration (positive integer or "all")
    "calib_selection": "random",  # selection method if more samples than <n_samples> in ["firsts", "random"]

    "w_interval": [-30, -18],  # manipulation prior window before grasp/release (<= 0)
    "thresh_physical_head": [30., 30., 30., 30.],  # threshold on head-target distance for: yaw+, yaw-, ekevation+, elevation- in degree
    "thresh_physical_gaze": [90., 90., 90., 90.],  # threshold on gaze-target distance for: yaw+, yaw-, ekevation+, elevation- in degree

    "calib_model": "linearHeadGaze",  # in ["none", "constant", "linearGaze", "linearHeadGaze"]
    "weight_calib_samples": True,  # boolean
    "ransac_n_iter": 500,
    "linearRidge_mu0": [1, 0, 0, 0, 0],  # default value for: gaze_scaling, gaze_shearing, head_scaling, head_shearing, translation
    "linearRidge_Lambda": [1e4, 1e4, 1e4, 1e4, 0],  # penalization for: gaze_scaling, gaze_shearing, head_scaling, head_shearing, translation
    "knn_n": 10,  # default number of reference points stored by KNN
    "knn_k": 3,  # default number of neighbours considered by KNN

    "thresh_vfoa": 10,  # in degree

    "online_n_min": 10,  # in number of frames
    "online_n_max": 1000,  # in number of frames
}


def stack_in_dict(dict, key, data_new):
    if key in dict.keys() and dict[key] is not None:
        dict[key] = np.vstack([dict[key], data_new])
    else:
        dict[key] = np.array(data_new)


def load_config(config_json=None, config_default=DEFAULT_CONFIG):
    """ return a config by mixing a default config and adding what is found in the input json (if given)
        <default_config>: (dict) dictonary containing configuration information
        <config_json>: (str) filepath of a json containgin configuration fileds to change/add """
    config = config_default

    if config_json is not None and os.path.exists(config_json):
        with open(config_json,) as f:
            config_from_json = json.load(f)
        config.update(config_from_json)
        # for key, value in config_from_json.items():
        #     config[key] = value

    return config


def load_data(data_files, only_annotated=False):
    """ load each file in <data_files> and return a dictionary of two dictionaries with subject's and targets' data respectively.
        Note that <data_targets> values contain both the 3D target position and the speaking status of the targets in a (-1,4) siye array
        <data_files>: (list) a list of filepath
        <only_annotated>: (bool) if True, keep only samples with a valid VFOA annotation (i.e. a target name or "aversion")"""
    if isinstance(data_files, str):
        data_files = [data_files]
        
    data_subject = {}
    data_targets = {}
    total_len = 0
    for data_file in data_files:
        data_lines = np.loadtxt(data_file, dtype=str, delimiter=',')

        # Filter data to keep
        if only_annotated:
            data_lines = data_lines[data_lines[:, 16] != 'na']
            data_lines = data_lines[data_lines[:, 16] != 'blink']
            data_lines = data_lines[data_lines[:, 16] != 'blinking']
            data_lines = data_lines[data_lines[:, 16] != 'no_data']
        
        # Subject's data
        stack_in_dict(data_subject, 'identifier', np.array(data_lines[:, 0], dtype=str).reshape((-1, 1)))
        stack_in_dict(data_subject, 'frameIndex', np.array(data_lines[:, 1], dtype=int).reshape((-1, 1)))
        stack_in_dict(data_subject, 'eye', np.array(data_lines[:, 2:5], dtype=float).reshape((-1, 3)))
        stack_in_dict(data_subject, 'gaze', np.array(data_lines[:, 5:8], dtype=float).reshape((-1, 3)))
        stack_in_dict(data_subject, 'head', np.array(data_lines[:, 8:11], dtype=float).reshape((-1, 3)))
        stack_in_dict(data_subject, 'headpose', np.array(data_lines[:, 11:14], dtype=float).reshape((-1, 3)))
        stack_in_dict(data_subject, 'speaking', np.array(data_lines[:, 14], dtype=int).reshape((-1, 1)))
        stack_in_dict(data_subject, 'action', np.array(data_lines[:, 15], dtype=str).reshape((-1, 1)))
        stack_in_dict(data_subject, 'vfoa_gt', np.array(data_lines[:, 16], dtype=str).reshape((-1, 1)))

        # Target's data
        target_list = data_lines[0][17:][::5]
        target_index = dict(zip(target_list, range(len(target_list))))
        for target_name in set(data_targets.keys()).union(target_list):
            if target_name in target_list:
                if target_name not in data_targets.keys():
                    data_targets[target_name] = np.nan * np.empty((total_len, 4))
                i = target_index[target_name]
                stack_in_dict(data_targets, target_name, np.array(data_lines[:, 17 + 5*i + 1: 17 + 5*(i+1)], dtype=float).reshape((-1, 4)))
            elif target_name in data_targets.keys():
                stack_in_dict(data_targets, target_name, np.nan * np.empty((len(data_lines), 4)))

        total_len += len(data_lines)
    
    data = {'subject': data_subject, 'targets': data_targets}
    return data


def save_data(data, output_filename, verbose=True):
    """ <data>: (dict) data structure, like the one outputed by the "load_data" function
        <output_filename>: (str) name of the file where to save data """
    n_targets = len(data['targets'].keys())

    # Generate header
    header = 'id,frameIndex,eye.x,eye.y,eye.z,gaze.x,gaze.y,gaze.z,head.x,head.y,head.z,headpose.roll,headpose.pitch,headpose.yaw,speaking,action,vfoa_GT'
    for i, target in enumerate(data['targets'].keys()):
        tar = 'target{}'.format(i)
        header += ',{}.name,{}.x,{}.y,{}.z,{}.speaking'.format(tar, tar, tar, tar, tar)

    # Generate data table
    data_table = np.array([]).reshape((0, 17 + 5*n_targets))
    for i in range(len(data['subject']['frameIndex'])):
        data_line = []
        for key in ['identifier', 'frameIndex', 'eye', 'gaze', 'head', 'headpose', 'speaking', 'action', 'vfoa_gt']:
            data_line.extend(data['subject'][key][i])
        for key in data['targets'].keys():
            data_line.append(key)
            data_line.extend(data['targets'][key][i])

        data_table = np.vstack([data_table, data_line])

    np.savetxt(output_filename, data_table, fmt='%s', delimiter=',', header=header)
    if verbose:
        print('Data with calibrated gaze stored in {}'.format(output_filename))
