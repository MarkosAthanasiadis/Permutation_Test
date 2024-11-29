# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

import numpy as np

def compute_permutation_results(sig_res):
    """
    Compute the results of a permutation test.

    This function calculates the p-value, the z-scored Correct Classification Rate (ccr), 
    and the z-scored ccr bin vector based on the results from the permutation test.

    Parameters:
    -----------
    sig_res : dict
        A dictionary holding the results of the permutation test. It should have the following structure:
        - 'real_data': Contains the real ccr values and bin_ccrs.
        - 'shuffle_<n>': Contains the shuffled ccr values and bin_ccrs for each permutation.

    Returns:
    --------
    sig_res : dict
        The input dictionary with an additional 'permutation_results' entry, containing:
        - 'pvalue': The p-value of the permutation test.
        - 'zccr': The z-scored ccr value.
        - 'binszccr': The z-scored ccr bin vector.

    """
    
    # Get the real mean ccr and real bin_ccr values from the real data
    real_ccr = np.round(np.mean(sig_res['real_data']['ccrs']), 3)
    real_bin_ccr = sig_res['real_data']['bin_ccrs']

    # Collect all shuffled ccr values and bin_ccr values
    shuffle_ccr = []
    shuffle_bin_ccr = []
    num_shuffles = len(sig_res) - 2  # Exclude 'real_data' and 'permutation_results'

    for n in range(1, num_shuffles + 1):
        shuffle_key = f'shuffle_{n}'
        shuffle_ccr.append(np.round(np.mean(sig_res[shuffle_key]['ccrs']), 3))
        shuffle_bin_ccr.append(sig_res[shuffle_key]['bin_ccrs'])

    # Flatten the shuffle bin ccr array for z-scoring
    shuffle_bin_ccr = np.concatenate(shuffle_bin_ccr, axis=0)

    # Compute the z-scored ccr per bin
    bin_std = np.std(shuffle_bin_ccr, axis=0)
    bin_mean = np.mean(shuffle_bin_ccr, axis=0)
    
    # Avoid division by zero in z-scoring
    binzccr = (real_bin_ccr - bin_mean) / bin_std
    binzccr = [x for x in binzccr[0, :] if not np.isnan(x)]

    # Compute the p-value for the real ccr
    pvalue = np.mean([1 for x in shuffle_ccr if real_ccr < x])

    # Compute the z-scored ccr value
    mean_shuffle_ccr = np.mean(shuffle_ccr)
    std_shuffle_ccr = np.std(shuffle_ccr)
    
    if std_shuffle_ccr != 0:
        zccr = (real_ccr - mean_shuffle_ccr) / std_shuffle_ccr
    else:
        zccr = np.nan

    # Store results in the 'permutation_results' dictionary
    sig_res['permutation_results'] = {
        'pvalue': pvalue,
        'zccr': zccr,
        'binszccr': binzccr
    }

    return sig_res
