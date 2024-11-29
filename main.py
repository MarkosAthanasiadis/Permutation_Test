#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

# /significance/main.py
"""
Main Significance Test Logic
____________________________

This module contains the primary logic for running a permutation test. 
It handles data preparation, shuffling, and computation of results.

"""

import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from utils import shuffle_labels as shlab
from utils import model_functions as md
from utils import pvalue as pval


def main(sig_par):
    """
    Main function for running the permutation test.

    Args:
        sig_par (dict): Dictionary containing input data, parameters, and settings.

    Returns:
        dict: Results dictionary including CCRs, bin CCRs, and significance metrics.
    """

    # Extract parameters
    params       = sig_par['parameters']
    subsamplings = params['nsubsamplings']
    model_flag   = params['model_flag']
    lr, epochs   = params['lr_epochs']
    nshuffles    = params['nshuffles']
    shuffletype  = params['shuffle_type']

    x_data              = sig_par['original_data']
    y_labels            = sig_par['original_labels']
    trial_labels        = sig_par['trial_labels']
    trial_contributions = sig_par['trial_contributions']
    setcores            = sig_par['computation_cores']

    # Create permutation labels
    if shuffletype == 'block':
        y_labels, nshuffles = shlab.block(y_labels, nshuffles, trial_labels, trial_contributions)
    elif shuffletype == 'full':
        y_labels, nshuffles = shlab.full(y_labels, nshuffles)
    else:
        raise ValueError("Invalid shuffle type specified.")

    # Update nshuffles in parameters
    params['nshuffles'] = nshuffles

    # Split tasks across cores
    num_cores = min(setcores if setcores > 0 else multiprocessing.cpu_count(), multiprocessing.cpu_count())
    runs = range(nshuffles + 1)

    # Parallel execution of model training and testing
    ccrs = Parallel(n_jobs=num_cores, timeout=None)(
        delayed(md.train_test_model)(np.copy(x_data), np.copy(y_labels[nperm]), x_data.shape[1], lr, epochs, subsamplings, model_flag)
        for nperm in runs
    )

    # Aggregate results
    ccr_data = {'parameters': params}
    for nsh, result in enumerate(ccrs):
        key = 'real_data' if nsh == 0 else f'shuffle_{nsh}'
        ccr_data[key] = {'ccrs': result[0], 'bin_ccrs': result[1]}

    # Compute significance metrics
    ccr_data = pval.compute_permutation_results(ccr_data)

    return ccr_data
