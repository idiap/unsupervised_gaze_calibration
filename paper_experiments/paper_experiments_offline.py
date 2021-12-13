'''
This script allows to run a set of experiments as reported in the related paper.

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

from offline_calibration import offline_calibration_experiment
from src.data_loader import load_config, DEFAULT_CONFIG


def experiments_offline(data_files_calib, data_files_eval, config, prior_list, model_list,
                        output_file=None, only_annotated_samples=False, verbose=False):
    angErr_dict, vfoaAcc_dict = {}, {}
    for prior in prior_list:
        for model in model_list:
            for reg in [False, True] if 'linear' in model else [False]:
                experiment = '{}_{}{}'.format(prior, model, '_reg' if reg else '')
                if experiment not in angErr_dict.keys():
                    angErr_dict[experiment] = []
                    vfoaAcc_dict[experiment] = []

                # Set config
                config['calib_prior'] = prior
                config['calib_model'] = model
                config['linearRidge_Lambda'] = [1e4, 1e4, 1e4, 1e4, 0] if reg else [0, 0, 0, 0, 0]

                # Set output
                output_file_exp = None
                if output_file is not None:
                    output_file_exp = output_file.replace('.txt', '{}.txt'.format(experiment))

                # Run experiment
                angErr, vfoaAcc, _ = offline_calibration_experiment(data_files_calib, data_files_eval,
                                                                    config_json=config,
                                                                    output_calibrated_data_file=output_file_exp,
                                                                    only_annotated_samples=only_annotated_samples)
                angErr_dict[experiment].append(angErr)
                vfoaAcc_dict[experiment].append(vfoaAcc)
                if verbose:
                    print('\t{}\tangErr={:.3f}\tvfoaAcc={:.3f}'.format(experiment, angErr, vfoaAcc))
    
    return angErr_dict, vfoaAcc_dict


def experiments_offline_ubimpressed(data_folder, config_json=None, data_calibrated_folder=None, verbose=False):
    # Set config here instead of during experiment so we can quickly change parameters
    config = load_config(config_json) if config_json is not None else DEFAULT_CONFIG.copy()

    prior_list = ['oracle', 'speaking']
    model_list = ['none', 'constant', 'linearGaze', 'linearHeadGaze']
    
    angErr_dict, vfoaAcc_dict = {}, {}
    for data_file in os.listdir(data_folder):
        if 'UBImpressed' in data_file:
            data_files_calib = os.path.join(data_folder, data_file)
            data_files_eval = data_files_calib
            print('process {}'.format(data_files_eval))
            
            # Process
            output_file = os.path.join(data_calibrated_folder, data_file) if data_calibrated_folder is not None else None
            angErr_dict_tmp, vfoaAcc_dict_tmp = experiments_offline([data_files_calib], [data_files_eval], config,
                                                                    prior_list, model_list,
                                                                    output_file=output_file,
                                                                    only_annotated_samples=False,
                                                                    verbose=verbose)
            for key in angErr_dict_tmp.keys():
                if key in angErr_dict.keys():
                    angErr_dict[key] +=  angErr_dict_tmp[key]
                    vfoaAcc_dict[key] += vfoaAcc_dict_tmp[key]
                else:
                    angErr_dict[key] = angErr_dict_tmp[key]
                    vfoaAcc_dict[key] = vfoaAcc_dict_tmp[key]
                
    return angErr_dict, vfoaAcc_dict


def experiments_offline_kth(data_folder, config_json=None, data_calibrated_folder=None, verbose=False):
    # Set config here instead of during experiment so we can quickly change parameters
    config = load_config(config_json) if config_json is not None else DEFAULT_CONFIG.copy()

    prior_list = ['oracle', 'speaking']
    model_list = ['none', 'constant', 'linearGaze', 'linearHeadGaze']

    individual_parts = False
    
    angErr_dict, vfoaAcc_dict = {}, {}
    for data_file in os.listdir(data_folder):
        if 'KTHIdiap' in data_file: 
            if not individual_parts:
                if '_1.txt' not in data_file:  # Process only one time each subject on all sections
                    continue
                # Get all files that belong to the same subject
                data_files_calib = [os.path.join(data_folder, data_file)]
                for section in [2, 3, 4]:
                    path_tmp = data_files_calib[0].replace('_1.txt', '_{}.txt'.format(section))
                    if os.path.exists(path_tmp):
                        data_files_calib.append(path_tmp)
                data_files_eval = data_files_calib
                print('process {}'.format(data_files_eval))
            else:  # process each part individually
                if '_1.txt' in data_file:  # Use only part 2-3-4 for comparison with train/test protocol
                    continue
                data_files_calib = [os.path.join(data_folder, data_file)]
                data_files_eval = list(data_files_calib)
                print('process {}'.format(data_files_eval))
            
            # Process
            output_file = os.path.join(data_calibrated_folder, data_file.replace('_1.txt', 'txt')) if data_calibrated_folder is not None else None
            angErr_dict_tmp, vfoaAcc_dict_tmp = experiments_offline(data_files_calib, data_files_eval, config,
                                                                    prior_list, model_list,
                                                                    output_file=output_file,
                                                                    only_annotated_samples=False,
                                                                    verbose=verbose)
            for key in angErr_dict_tmp.keys():
                if key in angErr_dict.keys():
                    angErr_dict[key] +=  angErr_dict_tmp[key]
                    vfoaAcc_dict[key] += vfoaAcc_dict_tmp[key]
                else:
                    angErr_dict[key] = angErr_dict_tmp[key]
                    vfoaAcc_dict[key] = vfoaAcc_dict_tmp[key]

    return angErr_dict, vfoaAcc_dict


def experiments_offline_manigaze(session, data_folder, config_json=None, data_calibrated_folder=None, verbose=False):
    # Set config here instead of during experiment so we can quickly change parameters
    config = load_config(config_json) if config_json is not None else DEFAULT_CONFIG.copy()
    config['thresh_vfoa'] = 90
    
    model_list = ['none', 'constant', 'linearGaze', 'linearHeadGaze']
    prior_dict = {'intra': ['oracle'], 'cross': ['oracle'], 'unsup': ['manipulation']}
    
    angErr_dict, vfoaAcc_dict = {}, {}
    for data_file in os.listdir(data_folder):
        if 'ManiGaze' in data_file and session in data_file:
            if session == 'MT':
                session_eval = 'MT'
                session_calib = {'intra': 'MT', 'cross': 'ET_center', 'unsup': 'OM1'}
            elif session == 'ET':
                session_eval = 'ET_center'
                session_calib = {'intra': 'ET_center', 'cross': 'MT', 'unsup': 'OM1'}
            else:
                continue
            
            for exp in session_calib.keys():
                data_files_eval = os.path.join(data_folder, data_file)
                data_files_calib = data_files_eval.replace(session_eval, session_calib[exp])
                print('process {} (calib: {})'.format(data_files_eval, exp))

                # Process
                prior_list = prior_dict[exp]
                output_file = os.path.join(data_calibrated_folder, data_file) if data_calibrated_folder is not None else None
                angErr_dict_tmp, vfoaAcc_dict_tmp = experiments_offline([data_files_calib], [data_files_eval],
                                                                        config,
                                                                        prior_list, model_list,
                                                                        output_file=output_file,
                                                                        only_annotated_samples=False,
                                                                        verbose=verbose)
                for key in angErr_dict_tmp.keys():
                    key_full = '{}_{}'.format(exp, key)
                    if key_full in angErr_dict.keys():
                        angErr_dict[key_full] = np.hstack([angErr_dict[key_full], angErr_dict_tmp[key]])
                        vfoaAcc_dict[key_full] = np.hstack([vfoaAcc_dict[key_full], vfoaAcc_dict_tmp[key]])
                    else:
                        angErr_dict[key_full] = angErr_dict_tmp[key]
                        vfoaAcc_dict[key_full] = vfoaAcc_dict_tmp[key]

    return angErr_dict, vfoaAcc_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-dset', required=True)
    parser.add_argument('--data_folder', '-data', required=True)
    parser.add_argument('--data_calibrated_folder', '-data_out', default=None)
    parser.add_argument('--config_json', '-config', default=None,
                        help='json file containing same keys as DEFAULT_CONFIG')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if args.dataset.lower() in ['ubimpressed', 'ubi']:
        args.dataset = 'UBImpressed'
        angErr_dict, vfoaAcc_dict = experiments_offline_ubimpressed(args.data_folder, args.config_json,
                                                                    args.data_calibrated_folder, args.verbose)
    elif args.dataset.lower() in ['kth-idiap', 'kthidiap', 'kth']:
        args.dataset = 'KTH-Idiap'
        angErr_dict, vfoaAcc_dict = experiments_offline_kth(args.data_folder, args.config_json,
                                                            args.data_calibrated_folder, args.verbose)
    elif args.dataset.lower() in ['manigaze_et', 'manigaze_mt']:
        session = args.dataset.split('_')[1].upper()
        args.dataset = 'ManiGaze'
        angErr_dict, vfoaAcc_dict = experiments_offline_manigaze(session, args.data_folder, args.config_json,
                                                                 args.data_calibrated_folder, args.verbose)
    else:
        raise Exception('Unknown dataset: {}'.format(args.dataset))
    
    print('{} results (experiment, angErr, vfoaAcc):'.format(args.dataset))
    for experiment in sorted(angErr_dict.keys()):
        print('{}\t{:.3f}\t{:.3f}'.format(experiment, np.nanmean(angErr_dict[experiment]),
                                          np.nanmean(vfoaAcc_dict[experiment])).expandtabs(28))
        
