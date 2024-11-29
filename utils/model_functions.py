# -*- coding: utf-8 -*-
"""
@author: Markos Athanasiadis
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNet(nn.Module):
    """
    A simple linear neural network with one fully connected layer.
    The output is two nodes corresponding to two classes.

    Attributes:
    - fc1: Fully connected layer (input -> output)
    """

    def __init__(self, dimensions: int):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(dimensions, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels based on the highest probability from the output logits.
        """
        pred = self.forward(x)
        return torch.argmax(pred, dim=1)

class NonLinearNet(nn.Module):
    """
    A simple non-linear neural network with one hidden layer and Tanh activation.

    Attributes:
    - fc1: First fully connected layer (input -> hidden)
    - fc2: Second fully connected layer (hidden -> output)
    """

    def __init__(self, dimensions: int):
        super(NonLinearNet, self).__init__()
        self.fc1 = nn.Linear(dimensions, dimensions + 1)
        self.fc2 = nn.Linear(dimensions + 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels based on the highest probability from the output logits.
        """
        pred = self.forward(x)
        return torch.argmax(pred, dim=1)

def subsampling_perms(labindices, traintarget):
    """
    Generate subsampling permutations for a given set of indices for training and testing.

    Parameters:
    - labindices: List of indices for a specific label.
    - traintarget: Target size for the training set.

    Returns:
    - train_perms: Indices selected for training.
    - test_perms: Indices selected for testing.
    """
    perm = np.random.permutation(labindices)
    train_perms = perm[:traintarget]
    test_perms = perm[traintarget:]
    return list(train_perms), list(test_perms)

def train_test_model(x_dt,y_lb,dimensions,lr,epochs,subsamplings,model_flag):
    """
    Train and evaluate a neural network model with subsampling.

    Parameters:
    - x_dt: Dataset with features.
    - y_lb: Labels corresponding to the dataset.
    - dimensions: Dimensionality of the input data.
    - lr: Learning rate for training.
    - epochs: Number of epochs for training.
    - subsamplings: Number of subsampling iterations.
    - model_flag: Type of model ('linear' or 'non-linear').

    Returns:
    - A list containing the mean CCR for all subsamplings and the CCR per pattern.
    """
    
    # Get the indices per label and their contribution % in train-test sets
    label0_indices = np.where(y_lb == 0)[0]
    label1_indices = np.where(y_lb == 1)[0]
    train_size = x_dt.shape[0] // 2
    # Get the ratio of each label in the original set
    label0_ratio = len(label0_indices) / len(y_lb)
    label1_ratio = len(label1_indices) / len(y_lb)
    # Get the target size for each label for the training-testing set
    label0_train_target = int(np.ceil(train_size * label0_ratio))
    label1_train_target = int(np.ceil(train_size * label1_ratio))
    
    ccrs    = []
    correct = np.zeros((subsamplings, x_dt.shape[0]))
    used    = np.zeros((subsamplings, x_dt.shape[0]))

    for nsub in range(subsamplings):
        # Generate train-test subsamples using permuted indices
        train0_perms, test0_perms = subsampling_perms(label0_indices, label0_train_target)
        train1_perms, test1_perms = subsampling_perms(label1_indices, label1_train_target)

        ind_train = train0_perms + train1_perms
        ind_test = test0_perms + test1_perms

        X_train = torch.tensor(x_dt[ind_train], dtype=torch.float32)
        X_test = torch.tensor(x_dt[ind_test], dtype=torch.float32)
        y_train = torch.tensor(y_lb[ind_train], dtype=torch.long)
        y_test = torch.tensor(y_lb[ind_test], dtype=torch.long)

        # Remove features with the same value across all examples
        valid_features = [i for i in range(dimensions) if len(np.unique(x_dt[:, i])) > 1]
        X_train = X_train[:, valid_features]
        X_test = X_test[:, valid_features]
        
        model = LinearNet(len(valid_features)) if model_flag == 'linear' else NonLinearNet(len(valid_features))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train the model
        model.train()
        for _ in range(epochs):
            #Clear previous gradients 
            optimizer.zero_grad()
            # Predict the output of a given input
            y_pred = model(X_train)
            # Compute the losses with the criterion and add it to the list
            loss = criterion(y_pred, y_train)
            #Compute gradients from backward propagation
            loss.backward()
            #Adjust the weights
            optimizer.step()

        # Evaluate the model
        model.eval()
        y_pred_class_test = model.predict(X_test)
        ccr = (y_test == y_pred_class_test).float().mean().item()
        ccrs.append(ccr)

        # Update the correct/used matrices for CCR per bin
        correct_indices = np.where(y_test == y_pred_class_test)[0]
        correct_bin_indices = [ind_test[x] for x in correct_indices]

        correct[nsub, correct_bin_indices] = 1
        used[nsub, ind_test] = 1

    # CCR per bin calculation
    correct = np.sum(correct, axis=0)
    used = np.sum(used, axis=0)
    bin_ccrs = np.divide(correct, used, where=used != 0)

    return [ccrs, bin_ccrs.reshape(1, -1)]

