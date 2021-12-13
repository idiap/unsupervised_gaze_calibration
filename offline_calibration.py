'''
This script allows to run a gaze calibration experiment with the 'offline' protocol,
given a set of calibration and test data files as well as a (optional) configuration json file.

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
from src.select_samples_calib import get_samples_calib
from src.calibration import compute_calibration, calibrate_gaze
from src.estimation import compute_vfoa
from src.evaluation import compute_mean_angular_error, compute_vfoa_accuracy


def offline_calibration_experiment(data_files_calib, data_files_eval=None,
                                   config_json=None, output_log_file=None, output_calibrated_data_file=None, experiment_name=None,
                                   only_annotated_samples=False, verbose=False):
    if experiment_name is None and config_json is not None and isinstance(config_json, str):
        experiment_name = os.path.splitext(os.path.basename(config_json))[0]

    # Load data and parameters
    if verbose:
        print('Loading data (calibration)...')
    data_calib = load_data(data_files_calib, only_annotated=only_annotated_samples)
    data_eval = load_data(data_files_eval) if data_files_eval is not None else data_calib
    if isinstance(config_json, dict):
        config = config_json
    else:
        config = load_config(config_json=config_json)

    if verbose:
        print('\nLoaded data (calibration):')
        for key1 in sorted(data_calib.keys()):
            print('"{}":'.format(key1))
            for key2 in sorted(data_calib[key1].keys()):
                print('\t"{}", shape={}, sample[0]={}'.format(key2, data_calib[key1][key2].shape,
                                                              data_calib[key1][key2][0]))

        print('\nConfiguration:')
        for key in sorted(config.keys()):
            print('"{}": {}'.format(key, config[key]))

    # Calibration
    if verbose:
        print('\nCalibrate...')
    samples_calib = get_samples_calib(data_calib, config['calib_prior'],
                                      config['calib_n_samples'], config['calib_selection'],
                                      config['w_interval'],
                                      config['thresh_physical_head'], config['thresh_physical_gaze'])
    calib_parameters = compute_calibration(samples_calib, config['calib_model'], config['weight_calib_samples'],
                                           config['ransac_n_iter'],
                                           config['linearRidge_mu0'], config['linearRidge_Lambda'],
                                           config['knn_n'], config['knn_k'])

    if verbose:
        if len(samples_calib.keys()) > 0:
            print('\nCalibration samples ({} samples over {} in data):'
                  .format(len(samples_calib['identifier']), len(data_calib['subject']['identifier'])))
            for key, val in samples_calib.items():
                print('"{}", shape={}'.format(key, val.shape))

            print('\nVFOA')
            print('\tground truth: {}'.format(np.unique(samples_calib['vfoa_gt'], return_counts=True)))
            print('\tin all calibration data: {}'.format(np.unique(data_calib['subject']['vfoa_gt'], return_counts=True)))
        else:
            print('\nNo calibration samples (over {} samples in data)'
                  .format(len(data_calib['subject']['identifier'])))

        print('\nCalibration:')
        print('A={}, B={}, C={}'.format(calib_parameters[0].flatten(), calib_parameters[1].flatten(),
                                        calib_parameters[2].flatten()))
        
    # New estimations (calibrated gaze and VFOA)
    if verbose:
        print('\nRe-estimate gaze and VFOA...')
    gaze_calib = calibrate_gaze(data_eval['subject']['gaze'], data_eval['subject']['headpose'],
                                calib_parameters, config['calib_model'])
    vfoa = compute_vfoa(data_eval['subject']['eye'], gaze_calib, data_eval['targets'],
                        config['thresh_vfoa'])

    # Evaluate
    if verbose:
        print('\nEvaluate...')
    angErr_n_samples, angErr, _, _, _ = compute_mean_angular_error(gaze_calib,
                                                                   data_eval['subject']['eye'],
                                                                   data_eval['targets'],
                                                                   data_eval['subject']['vfoa_gt'])
    vfoaAcc_n_samples, vfoaAcc, vfoaConfMat = compute_vfoa_accuracy(vfoa,
                                                                    data_eval['subject']['vfoa_gt'],
                                                                    data_eval['targets'].keys())
    if verbose:
        print('\nEvaluation')
        print('angErr={} ({} samples)'.format(angErr, angErr_n_samples))
        print('vfoaAcc={} ({} samples)'.format(vfoaAcc, vfoaAcc_n_samples))
        print('vfoaAcc - confusion matrix=\n{}'.format(vfoaConfMat))
    
    # Write results
    now = datetime.datetime.now()
    if output_log_file is not None:
        with open(output_log_file, 'a') as f:
            log_line = '{}/{}/{}\t{}h{}'.format(now.day, now.month, now.year, now.hour, now.minute)
            log_line += '\t{}'.format(experiment_name if experiment_name is not None else 'default')
            log_line += '\t{}\t{}'.format(angErr, vfoaAcc)
            f.write(log_line + '\n')
        print('Results written in {}'.format(output_log_file))

    if output_calibrated_data_file is not None:
        data_eval['subject']['gaze'] = gaze_calib
        save_data(data_eval, output_filename=output_calibrated_data_file)
    
    return angErr, vfoaAcc, calib_parameters
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_files_calib', '-calib', nargs='+', required=True)
    parser.add_argument('--data_files_eval', '-eval', nargs='+', default=None)
    parser.add_argument('--config_json', '-config', default=None,
                        help='json file containing same keys as DEFAULT_CONFIG or already a dict')
    parser.add_argument('--output_log_file', '-out', default=None)
    parser.add_argument('--calibrated_output_data_file', '-calib_out', default=None)
    parser.add_argument('--only_annotated_samples', '-oa', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    _, _, _ = offline_calibration_experiment(args.data_files_calib, args.data_files_eval,
                                             args.config_json, args.output_log_file, args.calibrated_output_data_file,
                                             only_annotated_samples=args.only_annotated_samples, verbose=args.verbose)
