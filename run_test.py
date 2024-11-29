# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

"""
Significance Test Runner
________________________

This script runs a permutation test on a given dataset. It handles input parameters, 
validates data, and invokes the main computation logic from `main.py`.

"""

import time
import yaml
import pathlib
import pickle
from main import main
from utils.data_loader import load_and_prepare_data
from utils.timer import timer


# Prompt user for parameter file
param_file_name = input('Enter the parameters file name (e.g., parameters.yml):\n')

# Load parameters from YAML file
with open(param_file_name, 'r') as file:
    parameters = yaml.safe_load(file)

# Extract and process parameters from the loaded YAML file
setcores     = int(parameters['core_id'])
mainpath     = parameters['main_path']
data_name    = parameters['dataset_name']
subsamplings = int(parameters['subsamplings'])
nshuffles    = int(parameters['shuffle_count'])
shuffletype  = parameters['shuffle_id']
model_flag   = parameters['model_id']
lr           = float(parameters['learning_rate'])
epochs       = int(parameters['epochs'])

# Load data
x_data, y_labels, trial_labels, trial_contributions = load_and_prepare_data(mainpath, data_name)

# Check dataset validity
if x_data.shape[1] <= 1 or len([1 for x in y_labels if x == 1]) <= 2:
    raise ValueError('Not enough data patterns or dimensions. Use a different dataset.')

# Prepare parameters dictionary
sig_par = {
    'parameters': {
        'nsubsamplings': subsamplings,
        'model_flag': model_flag,
        'lr_epochs': [lr, epochs],
        'nshuffles': nshuffles,
        'shuffle_type': shuffletype
    },
    'original_data': x_data,
    'original_labels': y_labels,
    'trial_labels': trial_labels,
    'trial_contributions': trial_contributions,
    'computation_cores': setcores
}

# Run main permutation test
start_time = time.time()
ccr_results = main(sig_par)

# Save results
save_path = pathlib.Path(mainpath) / f"{data_name}_results" / "data_info"
save_path.mkdir(parents=True, exist_ok=True)
save_name = save_path / "permutation_test_results.pkl"
with open(save_name, 'wb') as f:
    pickle.dump(ccr_results, f)

# Log elapsed time
timer(start_time)






























