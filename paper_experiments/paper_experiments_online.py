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

from online_calibration import online_calibration_experiment
from src.data_loader import load_config, DEFAULT_CONFIG


def experiments_online(data_files, config,
                       prior_list, model_list, n_min_list, n_max_list,
                       verbose=False):
    angErr_dict, vfoaAcc_dict = {}, {}
    for prior in prior_list:
        for model in model_list:
            for n_min in n_min_list:
                for n_max in n_max_list:
                    experiment = '{}_{}_{}_{}'.format(prior, model, n_min, n_max)
                    if experiment not in angErr_dict.keys():
                        angErr_dict[experiment] = []
                        vfoaAcc_dict[experiment] = []

                    # Set config
                    config['calib_prior'] = prior
                    config['calib_model'] = model
                    config['online_n_min'] = n_min
                    config['online_n_max'] = n_max

                    # Run experiment
                    angErr, vfoaAcc = online_calibration_experiment(data_files, config_json=config, verbose=verbose)
                    angErr_dict[experiment].append(angErr)
                    vfoaAcc_dict[experiment].append(vfoaAcc)
                    if verbose:
                        print('\t{}\tangErr={:.3f}\tvfoaAcc={:.3f}'.format(experiment, angErr, vfoaAcc))
    
    return angErr_dict, vfoaAcc_dict


def experiments_online_ubimpressed(data_folder, config_json=None, verbose=False):
    # Set config here instead of during experiment so we can quickly change parameters
    config = load_config(config_json) if config_json is not None else DEFAULT_CONFIG.copy()

    prior_list = ['speaking']
    model_list = ['constant', 'linearHeadGaze']
    n_min_list = [10]
    n_max_list = [100, 1000, np.inf]
    
    angErr_dict, vfoaAcc_dict = {}, {}
    for data_file in os.listdir(data_folder):
        if 'UBImpressed' in data_file:
            data_files = os.path.join(data_folder, data_file)
            print('process {}'.format(data_files))
            
            # Process
            angErr_dict_tmp, vfoaAcc_dict_tmp = experiments_online([data_files], config,
                                                                   prior_list, model_list,
                                                                   n_min_list, n_max_list,
                                                                   verbose=verbose)
            for key in angErr_dict_tmp.keys():
                if key in angErr_dict.keys():
                    angErr_dict[key] +=  angErr_dict_tmp[key]
                    vfoaAcc_dict[key] += vfoaAcc_dict_tmp[key]
                else:
                    angErr_dict[key] = angErr_dict_tmp[key]
                    vfoaAcc_dict[key] = vfoaAcc_dict_tmp[key]
            break
    return angErr_dict, vfoaAcc_dict


def experiments_online_kth(data_folder, config_json=None, verbose=False):
    # Set config here instead of during experiment so we can quickly change parameters
    config = load_config(config_json) if config_json is not None else DEFAULT_CONFIG.copy()

    prior_list = ['speaking']
    model_list = ['constant', 'linearHeadGaze']
    n_min_list = [10]
    n_max_list = [100, 1000, np.inf]
    
    individual_parts = False
    
    angErr_dict, vfoaAcc_dict = {}, {}
    for data_file in os.listdir(data_folder):
        if 'KTHIdiap' in data_file: 
            if not individual_parts:
                if '_1.txt' not in data_file:  # Process only one time each subject on all sections
                    continue
                # Get all files that belong to the same subject
                data_files = [os.path.join(data_folder, data_file)]
                for section in [2, 3, 4]:
                    path_tmp = data_files[0].replace('_1.txt', '_{}.txt'.format(section))
                    if os.path.exists(path_tmp):
                        data_files.append(path_tmp)

            else:  # process each part individually
                if '_1.txt' in data_file:  # Use only part 2-3-4 for comparison with train/test protocol
                    continue
                data_files = [os.path.join(data_folder, data_file)]
            
            # Process
            print('process {}'.format(data_files))
            angErr_dict_tmp, vfoaAcc_dict_tmp = experiments_online(data_files, config,
                                                                   prior_list, model_list,
                                                                   n_min_list, n_max_list,
                                                                   verbose=verbose)
            for key in angErr_dict_tmp.keys():
                if key in angErr_dict.keys():
                    angErr_dict[key] +=  angErr_dict_tmp[key]
                    vfoaAcc_dict[key] += vfoaAcc_dict_tmp[key]
                else:
                    angErr_dict[key] = angErr_dict_tmp[key]
                    vfoaAcc_dict[key] = vfoaAcc_dict_tmp[key]

    return angErr_dict, vfoaAcc_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-dset', required=True)
    parser.add_argument('--data_folder', '-data', required=True)
    parser.add_argument('--config_json', '-config', default=None,
                        help='json file containing same keys as DEFAULT_CONFIG')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if args.dataset.lower() in ['ubimpressed', 'ubi']:
        args.dataset = 'UBImpressed'
        angErr_dict, vfoaAcc_dict = experiments_online_ubimpressed(args.data_folder, args.config_json, args.verbose)
    elif args.dataset.lower() in ['kth-idiap', 'kthidiap', 'kth']:
        args.dataset = 'KTH-Idiap'
        angErr_dict, vfoaAcc_dict = experiments_online_kth(args.data_folder, args.config_json, args.verbose)
    else:
        raise Exception('Unknown dataset: {}'.format(args.dataset))
    
    print('{} results (experiment, angErr, vfoaAcc):'.format(args.dataset))
    for experiment in sorted(angErr_dict.keys()):
        print('{}\t{:.3f}\t{:.3f}'.format(experiment, np.nanmean(angErr_dict[experiment]),
                                          np.nanmean(vfoaAcc_dict[experiment])).expandtabs(28))
        
