---
layout: post
section-type: post
title: Hyperparameter tuning in multitarget regression with scikit-learn
category: how-to
tags: [ridge regression, encoding model, scikit-learn]
comments: true
---
I recently wanted to fit a GLM with a large design matrix to a high-dimensional dataset
with few samples, as is typical in fMRI. Since the design matrix had a lot more predictors than observations, I naturally used a regularized regression model with integrated hyperparameter tuning, i.e. scikit-learn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html" target="_blank">RidgeCV</a>. Instead of tediously fitting a model and optimizing its parameters for each time series one after the other, I wanted to take advantage of the fact that <a href="https://scikit-learn.org/stable/modules/multiclass.html" target="_blank">all scikit-learn classifiers support multitarget classification/regression</a> by default. However, I noticed that RidgeCV finds the optimal hyperparameter across all the entire multitarget problem. This clearly isn't what I want because the time series could conceivably differ in terms of how noisy they are. 

The way I solved this problem was to use<a href="https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html" target="_blank">MultiOutputRegressor</a>. Also, it is very convenient if you want to parallelize the regression analysis on high-dimensional datasets (fMRI, M/EEG, etc.) using the n_job argument. Here is how I use it and a demonstration of how it differs from RidgeCV's inbuilt multitarget regression. The resulting figure shows in each row the variance explained for MultiOutputRegressor, RidgeCV, high penalty Ridge, and low penalty Ridge for two variables with different noise levels. Left column shows estimated weights plotted against true weights for a noisy variable, right column for a noise-free variable. 

Note that cross-validation in MultiOutputRegressor is done separately for each target variable and it correctly finds strong regularization to be optimal for the left variable, and weak regularization to be optimal for the right variable.

```python
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import explained_variance_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import PredefinedSplit
import matplotlib.pyplot as plt

n_y = 2
T = 600
n_features = 10
noise_gain = 5.
p_grid = [.1, 1., 10.]
weak_regularization = [np.min(p_grid)]
strong_regularization = [np.max(p_grid)]

np.random.seed(12345)

# true weights
b = 2. * np.random.rand(n_features, n_y) - 1

# design matrix
X = np.random.randn(T, n_features)

# target variables to predict
y = np.dot(X, b)

# add noise to first but not second variable
y[:, 0] += noise_gain * np.random.randn(T)
# Variable 1 should now favor strong regularization, variable should favor
# weak regularization.

# Separate into training and test sets: 2/3 and 1/3 of data respectively
X_train, y_train = X[:2 * T // 3, :], y[:2 * T // 3, :]
X_test, y_test = X[2 * T // 3:, :], y[2 * T // 3:, :]

# How cross-validation should be performed to find optimal regularization
# within training set only (hence "inner cv")
inner_fold_ids = np.kron(np.arange(2), np.ones(T // 3))
inner_cv = PredefinedSplit(inner_fold_ids)

# Define classifier using MultiOutputRegressor. It is easy to parallelize
# computations using the n_jobs argument
clf_multi = MultiOutputRegressor(RidgeCV(alphas=p_grid,
                                         scoring='explained_variance',
                                         cv=inner_cv),
                                 n_jobs=None)

# This classifier uses only the multitarget implementation of RidgeCV
clf_ridgecv = RidgeCV(alphas=p_grid, scoring='explained_variance',
                      cv=inner_cv)

# Classifiers with weak and strong regularization
clf_lo = RidgeCV(alphas=weak_regularization,
                 scoring='explained_variance', cv=inner_cv)
clf_hi = RidgeCV(alphas=strong_regularization,
                 scoring='explained_variance', cv=inner_cv)

clfs = [clf_multi, clf_ridgecv, clf_lo, clf_hi]
clf_names = ['multi', 'ridgecv', 'low alpha', 'high alpha']
_, ax = plt.subplots(len(clfs), n_y)
for clf_num, (name, clf) in enumerate(zip(clf_names, clfs)):
    # Fit each classifier to training data and obtain predictions
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    # Compare estimated parameters with truth
    for var_num in range(n_y):
        true_par = b[:, var_num]
        if name == 'multi':
            est_par = clf.estimators_[var_num].coef_
        else:
            est_par = clf.coef_[var_num]
        expvar = explained_variance_score(y_test[:, var_num], y_hat[:, var_num])
        ax[clf_num, var_num].scatter(true_par, est_par)
        ax[clf_num, var_num].set_xlabel('true model weights')
        ax[clf_num, var_num].set_ylabel('estimated')
        ax[clf_num, var_num].set_title('%s, exp var: %.06f' % (name, expvar))
plt.tight_layout()
plt.show()
```