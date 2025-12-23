import numpy as np
import os
import pickle
import seaborn as sbn
import scipy
import scipy.stats
import sklearn 
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, StratifiedGroupKFold, GroupShuffleSplit, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score


import OCD_symphony
from OCD_symphony.utils import proj_dir, today


def load_attn_T_stats():
    with open(os.path.join(proj_dir, 'results', 'attention_heads_20251202.pkl'), 'rb') as f:
        attn_heads = pickle.load(f)
    return attn_heads[0][0].statistic


def filter_attn_weights(attn, T, threshold) :
    """ Filters attention patterns to keep only significant one based on training set. 
    
    Parameters
    ----------
        attn: n_samples x n_roi x n_roi
            Training attention patterns

        threshold: float
            Statistical threshold to apply (T-stat)

    Return
    ------
        new_samples = n_samples x n_features

    """

    rows, cols = np.where(np.abs(T)>threshold)
    
    new_samples = []
    for i in range(attn.shape[0]):
        features = []
        for row,col in zip(rows, cols):
            features.append(attn[i,row,col])
        new_samples.append(np.array(features))

    return np.array(new_samples)
        

def T_filter_func(T, a=3, b=10):
    return 1 / (1 + np.exp(-a*np.abs(T)+b))


def preproc_input(X, params):
    X = filter_attn_weight(attn, T_stats, threshold)
    return X



def train_classifier(X, y, groups):
    # 1) Pipeline: scale -> elastic-net logistic regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=10000,
            l1_ratio=0.5,        # tuned via grid below
            C=1.0,               # tuned via grid below
        ))
    ])

    # 2) Hyperparam search with nested CV (small grid to keep it stable)
    param_grid = {
        "clf__C": [0.1, 0.3, 1.0, 3.0, 10.0],
        "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    # Search hyper-parameters
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
    search = GridSearchCV(pipe, param_grid, cv=list(inner.split(X, y, groups)), scoring="accuracy", n_jobs=32, verbose=1)
    search.fit(X, y, groups=groups)
    print(f"Best params: {search.best_params_}")

    # Re-train with best hyper-parameters
    outer = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0)
    scores = sklearn.model_selection.cross_validate(search.best_estimator_, X, y, groups=groups, cv=outer, n_jobs=32, verbose=0, return_train_score=True, return_estimator=True)
    print(f"CV Training accuracy \t\t = \t {scores['train_score'].mean():.3f} ± {scores['train_score'].std():.3f}")
    print(f"CV Test accuracy \t\t = \t {scores['test_score'].mean():.3f} ± {scores['test_score'].std():.3f}")

    return scores


def perform_classification(all_perceiver_attn, subjs):
    """ Main loop to perform a classification based on the attention weights.
    
    Parameters
    ----------
        all_perceiver_attn: torch.Tensor or numpy.array
            Attention weight (n_samples, n_heads, n_rois, n_rois)

        subjs: list
            Subject ids to create labels and groups
    
    Return 
    ------
        None
    
    """ 

    # create inpputs and labels
    X = all_perceiver_attn[:,0,:,:].squeeze()
    y = np.array([0 if 'control' in s else 1 for s in subjs])

    # create groups
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(subjs)
    groups = le.transform(subjs)

    final = sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0) 
    splits = final.split(X, y, groups)

    # param
    #threshold = 4
    
    # Loop for cross-validation (i.e. rotating 80% training set, 20% test set)
    for i,(train_inds,test_inds) in enumerate(splits):
        print(f"--- SPLIT {i+1} ---")
        print("Compute T-stat from training set...")
        ttest_res = OCD_symphony.attention_analysis.compute_weights_ttest(all_perceiver_attn[train_inds], y[train_inds])

        print("Perform dimensionality reduction on training set...")
        #X_train = OCD_symphony.predictive_analysis.filter_attn_weights(X[train_inds], T=ttest_res.statistic, threshold=4)
        filt = T_filter_func(ttest_res.statistic)[np.newaxis, :]
        pca = sklearn.decomposition.PCA(n_components=100)
        X_train  = X[train_inds] * filt
        X_train = X_train.reshape(-1, 450*450)
        X_train = pca.fit_transform(X_train)
        
        scores = OCD_symphony.predictive_analysis.train_classifier(X_train, y[train_inds], groups[train_inds])
        print("Run inference on test set...")
        for j,estimator in enumerate(scores['estimator']):
            #X_test = OCD_symphony.predictive_analysis.filter_attn_weights(X[test_inds], T=ttest_res.statistic, threshold=4)
            X_test = X[test_inds] * filt
            X_test = pca.transform(X_test.reshape(-1,450*450))
            y_pred = estimator.predict(X_test)
            accuracy = sklearn.metrics.accuracy_score(y[test_inds], y_pred)
            f1_score = sklearn.metrics.f1_score(y[test_inds], y_pred)
            print(f"\tEstimator {j+1}: \t Accuracy = {accuracy:.2f} \t F1 = {f1_score:.2f}")