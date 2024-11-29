# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

import numpy as np
import itertools

def block(y_labels, nshuffles, label_values, contribution):
    """
    Shuffle labels in a block/trial manner.

    This function shuffles labels within blocks (trials) and adjusts 
    the number of shuffles based on the number of trials.

    Parameters:
    - y_labels (numpy array): The array of labels to be shuffled.
    - nshuffles (int): The number of shuffles to perform.
    - label_values (list): The values associated with each label for each trial.
    - contribution (list): The contribution of each trial (length of the trial).

    Returns:
    - y_perms (list of numpy arrays): A list of shuffled labels for each shuffle.
    - nshuffles (int): The adjusted number of shuffles, taking into account 
      the factorial constraint for small numbers of trials.
    """

    if not isinstance(y_labels, np.ndarray):
        raise ValueError("y_labels must be a numpy array.")
    if len(label_values) != len(contribution):
        raise ValueError("label_values and contribution must have the same length.")
    
    # Number of trials (indices of the label values)
    trials = np.arange(len(label_values))
    
    # Adjust the number of shuffles if the number of trials is small
    if len(trials) <= 6:
        new_nshuffles = np.math.factorial(len(trials))
        nshuffles = min(nshuffles, new_nshuffles)
    
    # Generate permutations
    perms = [trials]  # Start with the original order (unshuffled)
    
    # If not all possible permutations are needed, generate random unique permutations
    if nshuffles != np.math.factorial(len(trials)):
        while len(perms) <= nshuffles:
            perm_i = np.random.permutation(trials)  # Random permutation
            if not any(np.array_equal(perm_i, p) for p in perms):  # Ensure uniqueness
                perms.append(perm_i)
    else:
        perms.extend(itertools.permutations(trials))  # Use all permutations if required
    
    # Create a list to store the shuffled labels for each shuffle
    y_perms = []
    
    # Loop through each shuffle
    for perm in perms[:nshuffles+1]:
        labels = []
        
        # Construct the shuffled labels per trial based on the permutation
        for ntr, trial_idx in enumerate(perm):
            cont = contribution[ntr]
            trial_labels = np.zeros(cont)
            trial_labels[:] = 1 if label_values[trial_idx] == 1 else 0
            
            labels.append(trial_labels)
        
        # Concatenate labels across trials for this shuffle
        y_perms.append(np.concatenate(labels, axis=0))
    
    return y_perms, nshuffles


def full(y_labels, nshuffles):
    """
    Shuffle labels in a full/pattern manner.

    This function shuffles the labels as a full block (pattern) and 
    adjusts the number of shuffles based on the number of labels.

    Parameters:
    - y_labels (numpy array): The array of labels to be shuffled.
    - nshuffles (int): The number of shuffles to perform.

    Returns:
    - y_perms (list of numpy arrays): A list of shuffled labels for each shuffle.
    - nshuffles (int): The adjusted number of shuffles, considering the factorial 
      constraint for small numbers of labels.
    """

    if not isinstance(y_labels, np.ndarray):
        raise ValueError("y_labels must be a numpy array.")
    
    # Number of labels (indices of the label array)
    indices = np.arange(len(y_labels))
    
    # Adjust the number of shuffles if the number of labels is small
    if len(indices) <= 6:
        new_nshuffles = np.math.factorial(len(indices))
        nshuffles = min(nshuffles, new_nshuffles)
    
    # Generate random permutations while ensuring uniqueness
    perms = [list(indices)]  # Start with the original (unshuffled) order
    while len(perms) <= nshuffles:
        perm_i = np.random.permutation(indices)
        if not any(np.array_equal(perm_i, p) for p in perms):
            perms.append(list(perm_i))
    
    # Create a list to store the shuffled labels for each shuffle
    y_perms = []
    
    # Loop through each shuffle
    for perm in perms[:nshuffles+1]:
        # Get the shuffled labels for this permutation
        shuffled_labels = y_labels[perm]
        
        # Adjust the labels (replace -1 with 0, and 1 remains 1)
        adjusted_labels = np.where(shuffled_labels == 1, 1, 0)
        
        y_perms.append(adjusted_labels)
    
    return y_perms, nshuffles



