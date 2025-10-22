import argparse
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sbn
import scipy
import sklearn
import torch

import OCD_symphony
from OCD_symphony.utils import proj_dir, today


def classify(estimator: sklearn.base.BaseEstimator,
             X: np.array, y: np.array, groups: np.array, 
             #X_train: np.Array, y_train: np.Array, 
             #X_test: np.Array, y_test: np.Array,
             #CV:  = 5, 
             args:argparse.Namespace = None) -> dict:
    
    """ Template estimator for regression/classification """

    # get splits
    cv = sklearn.model_selection.StratifiedGroupKFold(n_splits=args.n_splits)
    #split_train_inds, split_test_inds = cv.split(X,y,groups)

    # loop over cv folds
    output = {'estimator':estimator, 'scores':[]}
    for train_inds, test_inds in cv.split(X,y,groups):
        
        # train
        estimator.fit(X[train_inds], y[train_inds])

        # test / score
        output['scores'].append(estimator.score(X[test_inds], y[test_inds]))

    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', action='store', default=None, help='subject to process')
    parser.add_argument('--subdir', action='store', default='', help='Sub-directory in proj_dir/data/ to save the torch files')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()