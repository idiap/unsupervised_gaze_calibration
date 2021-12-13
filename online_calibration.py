'''
This script allows to run a gaze calibration experiment with the 'online' protocol,
given a set of data files as well as a (optional) configuration json file.

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
import datetime
import numpy as np
import json

from src.data_loader import load_data, load_config, save_data
from src.select_samples_calib import check_physical_constraints
from src.calibration import compute_calibration, calibrate_gaze
from src.estimation import compute_vfoa_sample
from src.evaluation import compute_mean_angular_error, compute_vfoa_accuracy
from src.geometry import vectorToYawElevation


class OnlineCalibrationSet:
    def __init__(self, n_min=10, n_max=1000):
        self.n_min = n_min
        self.n_max = n_max
        self.calibration_set = {}

    def add_feature(self, key, data_new):
        if key in self.calibration_set.keys() and self.calibration_set[key] is not None:
            self.calibration_set[key] = np.vstack([self.calibration_set[key], data_new])
        else:
            self.calibration_set[key] = np.array(data_new)

    def update_calibration_set(self, gaze, headpose, target, target_name,
                               identifier=None, frameIndex=None, vfoa_gt=None):
        # Remove a sample at random if needed
        if self.get_length() >= self.n_max:
            indexes = range(self.get_length())
            np.random.shuffle(indexes)
            indexes = indexes[0:self.n_max-1]
            for key in self.calibration_set.keys():
                self.calibration_set[key] = self.calibration_set[key][indexes]

        # Add new sample
        self.add_feature('identifier', np.reshape(identifier, (-1, 1)))
        self.add_feature('frameIndex', np.reshape(frameIndex, (-1, 1)))
        self.add_feature('vfoa_gt', np.reshape(vfoa_gt, (-1, 1)))
        self.add_feature('gaze', np.reshape(gaze, (-1, 2)))
        self.add_feature('headpose', np.reshape(headpose, (-1, 2)))
        self.add_feature('target', np.reshape(target, (-1, 2)))
        self.add_feature('target_name', np.reshape([target_name], (-1, 1)))

    def get_calib_set(self):
        return self.calibration_set

    def get_length(self):
        if len(self.calibration_set.keys()) == 0:
            return 0
        else:
            return self.calibration_set['gaze'].shape[0]


def vfoa_labeling_oracle(eye, gaze, headpose, targets_dict, vfoa_gt):
    target_name = vfoa_gt
    if target_name in tarets_dict.keys():
        target_pos = targets_dict[target_name][0:3]
        if not np.isnan(target_pos).any():
            gaze_2D = vectorToYawElevation(gaze)
            target_2D = vectorToYawElevation(target_pos - eye)
            headpose_deg = np.array(headpose) * 180/np.pi
            headpose_CCS_2D = [headpose_deg[2], -headpose_deg[1]]
            return target_name, gaze_2D, headpose_CCS_2D, target_2D

    return 'aversion', None, None, None

                
def weak_vfoa_labeling_conversation(eye, gaze, headpose, speaking, targets_dict,
                                    thresh_physical_head=[30., 30., 30., 30.],
                                    thresh_physical_gaze=[90., 90., 90., 90.]):
    """ Similar to "src/select_samples_calib.py" process but for one single sample """

    # Get speaker
    speakers = []
    if speaking == 1:
        speakers.append('subject')
    for target_name, target_data in targets_dict.items():
        if target_data[3] == 1:
            speakers.append(target_name)
    target_name = speakers[0] if len(speakers) == 1 else None

    if target_name in targets_dict.keys():
        target_pos = targets_dict[target_name][0:3]
        if not np.isnan(target_pos).any() and not np.isnan(gaze).any():
            gaze_2D = vectorToYawElevation(gaze)
            target_2D = vectorToYawElevation(target_pos - eye)
            
            # Headpose is (0,0) as we work in Head Coordinate System.
            # Still store real headpose, as it can be useful in calibration model
            headpose_CCS_deg = np.array(headpose) * 180/np.pi
            headpose_CCS_2D = [headpose_CCS_deg[2], -headpose_CCS_deg[1]]
            headpose_HCS = [0., 0.]
                
            # Check physical constraints
            if check_physical_constraints(headpose_HCS, target_2D, thresh_physical_head) and \
               check_physical_constraints(gaze_2D, target_2D, thresh_physical_gaze):
                return target_name, gaze_2D, headpose_CCS_2D, target_2D

    return 'aversion', None, None, None

    
def online_calibration_experiment(data_files, config_json=None,
                                  output_log_file=None, output_calibrated_data_file=None,
                                  experiment_name=None, only_annotated_samples=False, verbose=False):
    if experiment_name is None and config_json is not None and isinstance(config_json, str):
        experiment_name = os.path.splitext(os.path.basename(config_json))[0]

    # Load data and parameters
    data = load_data(data_files, only_annotated=only_annotated_samples)
    config = config_json if isinstance(config_json, dict) else load_config(config_json=config_json)

    if verbose:
        print('\nConfiguration:')
        for key in sorted(config.keys()):
            print('"{}": {}'.format(key, config[key]))

    # Init
    online_calib_set = OnlineCalibrationSet(n_min=config['online_n_min'], n_max=config['online_n_max'])
    calib_params = None
    gaze_calib, vfoa_calib = np.array([]).reshape((0, 3)), np.array([]).reshape((0, 1))
    vfoa_baseline = np.array([]).reshape((0, 1))
    
    # Run
    for i in range(data['subject']['gaze'].shape[0]):

        # build targets dictionary for this sample
        targets_dict = {}
        for target_name, target_data in data['targets'].items():
            targets_dict[target_name] = target_data[i]
        
        # Weak VFOA labeling (get vfoa weak label as well as 2D gaze, headpose, and target position)
        if config['calib_prior'] == 'oracle':
            target_name, gaze, headpose, target = vfoa_labeling_oracle(data['subject']['eye'][i],
                                                                       data['subject']['gaze'][i],
                                                                       data['subject']['headpose'][i],
                                                                       targets_dict,
                                                                       data['subject']['vfoa_gt'][i])
        elif config['calib_prior'] == 'speaking':
            target_name, gaze, headpose, target = weak_vfoa_labeling_conversation(data['subject']['eye'][i],
                                                                                  data['subject']['gaze'][i],
                                                                                  data['subject']['headpose'][i],
                                                                                  data['subject']['speaking'][i],
                                                                                  targets_dict,
                                                                                  config['thresh_physical_head'],
                                                                                  config['thresh_physical_gaze'])
        else:
            raise Exception('Unsuported prior: {}'.format(config['calib_prior']))

        # Update calibration set (if we found a not 'aversion' sample)
        if target_name in targets_dict.keys():
            online_calib_set.update_calibration_set(gaze, headpose, target, target_name,
                                                    data['subject']['identifier'][i],
                                                    data['subject']['frameIndex'][i],
                                                    data['subject']['vfoa_gt'][i])
        
            # Update calibration parameters (because calibration set changed)
            if online_calib_set.get_length() >= config['online_n_min']:
                calib_params = compute_calibration(online_calib_set.get_calib_set(),
                                                   calib_model=config['calib_model'],
                                                   weight_calib_samples=True,
                                                   ransac_n_iter=config['ransac_n_iter'],
                                                   linearRidge_mu0=config['linearRidge_mu0'],
                                                   linearRidge_Lambda=config['linearRidge_Lambda'])
        
        # Calibrate gaze and get vfoa
        g_calib = calibrate_gaze(data['subject']['gaze'][[i]], data['subject']['headpose'][[i]], calib_params)
        v_calib = compute_vfoa_sample(data['subject']['eye'][i], g_calib, targets_dict, config['thresh_vfoa'])
        v_baseline = compute_vfoa_sample(data['subject']['eye'][i], data['subject']['gaze'][i], targets_dict, config['thresh_vfoa'])

        gaze_calib = np.vstack([gaze_calib, g_calib.reshape(-1, 3)])
        vfoa_calib = np.vstack([vfoa_calib, np.reshape([v_calib], (-1, 1))])
        vfoa_baseline = np.vstack([vfoa_baseline, np.reshape([v_baseline],(-1, 1))])
        
        # Display
        if verbose and i % 100 == 0:
            print('frame {}/{}: calibration_set_size={}'.format(i, data['subject']['gaze'].shape[0],
                                                                online_calib_set.get_length()))

    # Compute global metrics
    angErr_metrics = compute_mean_angular_error(data['subject']['gaze'], data['subject']['eye'],
                                                data['targets'], data['subject']['vfoa_gt'])
    _, angErr_3D_baseline, angErr_2D_baseline, _, _ = angErr_metrics

    angErr_metrics = compute_mean_angular_error(gaze_calib, data['subject']['eye'],
                                              data['targets'], data['subject']['vfoa_gt'])
    angErr_n_samples, angErr_3D_calib, angErr_2D_calib, _, _ = angErr_metrics

    _, vfoaAcc_baseline, _ = compute_vfoa_accuracy(vfoa_baseline, data['subject']['vfoa_gt'],
                                                                   data['targets'].keys())

    vfoaAcc_n_samples, vfoaAcc_calib, vfoaConfMat = compute_vfoa_accuracy(vfoa_calib, data['subject']['vfoa_gt'],
                                                                          data['targets'].keys())

    if verbose:
        print('\nEvaluation')
        print('angErr_3D={} ({} samples, before calib={})'.format(angErr_3D_calib, angErr_n_samples, angErr_3D_baseline))
        print('angErr_2D={} ({} samples, before calib={})'.format(angErr_2D_calib, angErr_n_samples, angErr_2D_baseline))
        print('vfoaAcc={} ({} samples, before_calib={})'.format(vfoaAcc_calib, vfoaAcc_n_samples, vfoaAcc_baseline))
        print('vfoaAcc - confusion matrix=\n{}'.format(vfoaConfMat))

    # Write results
    now = datetime.datetime.now()
    if output_log_file is not None:
        with open(output_log_file, 'a') as f:
            log_line = '{}/{}/{}\t{}h{}'.format(now.day, now.month, now.year, now.hour, now.minute)
            log_line += '\t{}'.format(experiment_name if experiment_name is not None else 'default')
            log_line += '\t{}\t{}'.format(angErr_3D_calib, vfoaAcc_calib)
            f.write(log_line + '\n')
        print('Results written in {}'.format(output_log_file))

    if output_calibrated_data_file is not None:
        data_eval['subject']['gaze'] = gaze_calib
        save_data(data_eval, output_filename=output_calibrated_data_file)

    return angErr_3D_calib, vfoaAcc_calib
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_files', '-data', nargs='+', required=True)
    parser.add_argument('--config_json', '-config', default=None,
                        help='json file containing same keys as DEFAULT_CONFIG or already a dict')
    parser.add_argument('--output_log_file', '-out', default=None)
    parser.add_argument('--calibrated_output_data_file', '-calib_out', default=None)
    parser.add_argument('--only_annotated_samples', '-oa', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    _, _ = online_calibration_experiment(args.data_files, args.config_json,
                                         args.output_log_file, args.calibrated_output_data_file,
                                         only_annotated_samples=args.only_annotated_samples, verbose=args.verbose)
