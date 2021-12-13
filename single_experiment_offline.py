'''
This script allows to run a single experiment with adaptation to each dataset proposed in the 
related paper.

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


def experiments_offline_ubimpressed(data_folder, config_json=None,
                                    prior='oracle', model='constant', reg=True,
                                    only_annotated=False, verbose=False):    
    # Set config here instead of during experiment so we can quickly change parameters
    config = load_config(config_json) if config_json is not None else DEFAULT_CONFIG.copy()

    config['calib_prior'] = prior
    config['calib_model'] = model
    config['linearRidge_Lambda'] = [1e4, 1e4, 1e4, 1e4, 0] if reg else [0, 0, 0, 0, 0]

    angErr_list, vfoaAcc_list = [], []
    for data_file in os.listdir(data_folder):
        if 'UBImpressed' in data_file:
            data_files_calib = os.path.join(data_folder, data_file)
            data_files_eval = data_files_calib
            print('process {}'.format(data_files_eval))
            
            # Process
            angErr, vfoaAcc, _ = offline_calibration_experiment(data_files_calib, data_files_eval,
                                                                config_json=config,
                                                                only_annotated_samples=only_annotated)
            angErr_list.append(angErr)
            vfoaAcc_list.append(vfoaAcc)
            
    experiment = 'ubi_{}_{}{}'.format(prior, model, '_reg' if reg else '')
    angErr_dict = {experiment: np.mean(angErr_list)}
    vfoaAcc_dict = {experiment: np.mean(vfoaAcc_list)}
    return angErr_dict, vfoaAcc_dict


def experiments_offline_kth(data_folder, config_json=None,
                            prior='oracle', model='constant', reg=True,
                            only_annotated=False, verbose=False):
    # Set config here instead of during experiment so we can quickly change parameters
    config = load_config(config_json) if config_json is not None else DEFAULT_CONFIG.copy()

    config['calib_prior'] = prior
    config['calib_model'] = model
    config['linearRidge_Lambda'] = [1e4, 1e4, 1e4, 1e4, 0] if reg else [0, 0, 0, 0, 0]

    individual_parts = False
    
    angErr_list, vfoaAcc_list = [], []
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
            angErr, vfoaAcc, _ = offline_calibration_experiment(data_files_calib, data_files_eval,
                                                                config_json=config,
                                                                only_annotated_samples=only_annotated)
            angErr_list.append(angErr)
            vfoaAcc_list.append(vfoaAcc)

    experiment = 'kth_{}_{}{}'.format(prior, model, '_reg' if reg else '')
    angErr_dict = {experiment: np.mean(angErr_list)}
    vfoaAcc_dict = {experiment: np.mean(vfoaAcc_list)}
    return angErr_dict, vfoaAcc_dict


def experiments_offline_manigaze(session, data_folder, config_json=None,
                                 model='constant', reg=True,
                                 only_annotated=False, verbose=False):
    # Set config here instead of during experiment so we can quickly change parameters
    config = load_config(config_json) if config_json is not None else DEFAULT_CONFIG.copy()
    config['thresh_vfoa'] = 90
    
    config['calib_model'] = model
    config['linearRidge_Lambda'] = [1e4, 1e4, 1e4, 1e4, 0] if reg else [0, 0, 0, 0, 0]

    prior_dict = {'intra': ['oracle'], 'cross': ['oracle'], 'unsup': ['manipulation']}
    if session == 'MT':
        session_eval = 'MT'
        session_calib = {'intra': 'MT', 'cross': 'ET_center', 'unsup': 'OM1'}
    elif session == 'ET':
        session_eval = 'ET_center'
        session_calib = {'intra': 'ET_center', 'cross': 'MT', 'unsup': 'OM1'}
    else:
        raise Exception('ManiGaze - unknown session {}'.format(session))

    angErr_dict, vfoaAcc_dict = {}, {}
    for exp in session_calib.keys():
        angErr_list, vfoaAcc_list = [], []
        for data_file in os.listdir(data_folder):
            if 'ManiGaze' in data_file and session in data_file:
                data_files_eval = os.path.join(data_folder, data_file)
                data_files_calib = data_files_eval.replace(session_eval, session_calib[exp])
                print('process {} (calib: {})'.format(data_files_eval, exp))

                # Process
                config['calib_prior'] = prior_dict[exp]
                angErr, vfoaAcc, _ = offline_calibration_experiment(data_files_calib, data_files_eval,
                                                                    config_json=config,
                                                                    only_annotated_samples=only_annotated)
                angErr_list.append(angErr)
                vfoaAcc_list.append(vfoaAcc)

        experiment = 'mg_{}_{}{}'.format(exp, model, '_reg' if reg else '')
        angErr_dict[experiment] = np.mean(angErr_list)
        vfoaAcc_dict[experiment] = np.mean(vfoaAcc_list)
            
    return angErr_dict, vfoaAcc_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-dset', required=True)
    parser.add_argument('--data_folder', '-data', required=True)
    parser.add_argument('--config_json', '-config', default=None,
                        help='json file containing same keys as DEFAULT_CONFIG')
    parser.add_argument('--prior', '-p', default='oracle')
    parser.add_argument('--model', '-m', default='constant')
    parser.add_argument('--noreg', '-nr', action='store_true')
    parser.add_argument('--only_annotated', '-oa', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if args.dataset.lower() in ['ubimpressed', 'ubi']:
        args.dataset = 'UBImpressed'
        angErr_dict, vfoaAcc_dict = experiments_offline_ubimpressed(args.data_folder, args.config_json,
                                                                    args.prior, args.model, not args.noreg,
                                                                    args.only_annotated, args.verbose)
    elif args.dataset.lower() in ['kth-idiap', 'kthidiap', 'kth']:
        args.dataset = 'KTH-Idiap'
        angErr_dict, vfoaAcc_dict = experiments_offline_kth(args.data_folder, args.config_json,
                                                            args.prior, args.model, not args.noreg,
                                                            args.only_annotated, args.verbose)
    elif args.dataset.lower() in ['manigaze_et', 'manigaze_mt']:
        session = args.dataset.split('_')[1].upper()
        args.dataset = 'ManiGaze'
        angErr, vfoaAcc = experiments_offline_manigaze(session, args.data_folder, args.config_json,
                                                       args.model, not args.noreg,
                                                       args.only_annotated, args.verbose)
    else:
        raise Exception('Unknown dataset: {}'.format(args.dataset))
    
    print('{} results (experiment, angErr, vfoaAcc):'.format(args.dataset))
    for experiment in sorted(angErr_dict.keys()):
        print('{}\t{:.3f}\t{:.3f}'.format(experiment, np.nanmean(angErr_dict[experiment]),
                                          np.nanmean(vfoaAcc_dict[experiment])).expandtabs(28))
