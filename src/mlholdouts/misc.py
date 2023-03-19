import os
import sys

import pandas as pd
import numpy as np

import pywt
import scipy.io as spio
from scipy.stats import entropy
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import timeit

import py_ddspls
import sklearn.metrics as sklm
import sklearn.preprocessing as skpre


# maximum lambda that can be used before returning a empty model
def get_max_lambda(X, Y):
    MMss0 = py_ddspls.model.ddspls(X, Y, lambd=0, R=1, mode="reg").model.Ms
    K = len(MMss0)
    lambd_max_w = 0
    for k in range(K):
        lambd_max_w = max([lambd_max_w, np.max(abs(MMss0[k]))])

    return lambd_max_w


from scipy.stats import pearsonr


# this for the training data
def get_r_value(yreal, ypred):
    mean = np.mean(yreal)
    num = np.sum(np.square(yreal - ypred))
    denom = np.sum(np.square(yreal - mean))
    R = 1 - (num / denom)
    return R


# this for the testing data
def get_q_value(yreal, ypred, train_mean):
    num = np.sum(np.square(yreal - ypred))
    denom = np.sum(np.square(yreal - train_mean))
    R = 1 - (num / denom)
    return R


def get_correlation(X, Y, model, method="ddspls"):
    corr = None
    if method == "ddspls":
        K = len(model.u)
        if K == 1:
            u = model.u[0]
            v = model.v
            Xu = np.dot(X, u)
            Yv = np.dot(Y, v)
            #
            corr, _ = pearsonr(Xu[:, 0], Yv[:, 0])
        else:
            Xu = np.zeros(shape=(Y.shape[0], 1), like=Y)
            for k in range(K):
                Xu = Xu + np.dot(X[k], model.u[k])

            v = model.v
            Yv = np.dot(Y, v)
            #
            corr, _ = pearsonr(Xu[:, 0], Yv[:, 0])
    elif (method == "svr") or (method == "lasso"):  # or (method == "ridge"):
        pred = model.predict(X)
        corr, _ = pearsonr(pred, Y)
    elif (method == "ridge"):
        pred = model.predict(X)
        corr = 1 - np.mean(np.abs(pred - Y) / Y)
    return corr


def get_pairwise_overlap(u, v, tol=1e-6):
    bool_u = (np.abs(u) <= tol)
    bool_v = (np.abs(v) <= tol)

    uv = np.logical_and(bool_u, bool_v)

    S1 = np.sum(bool_u) / u.shape[0]
    S2 = np.sum(bool_v) / v.shape[0]

    E = u.shape[0] * S1 * S2

    O = (np.sum(uv) - E) / (max(np.sum(bool_v), np.sum(bool_u)) + 1)

    return O


def get_stability(models, method="ddspls"):
    if method == "ddspls":
        n = len(models)
        all_overlap = []
        K = len(models[0].u)
        for idx_model_1 in range(n - 1):
            for idx_model_2 in range(idx_model_1 + 1, n):
                view_overlap = []
                for k in range(K):
                    view_overlap.append(get_pairwise_overlap(models[idx_model_1].u[k], models[idx_model_2].u[k]))
                all_overlap.append(np.mean(view_overlap))

        score = 2 * np.sum(all_overlap) / (n * (n - 1))
    elif (method == "svr"):
        n = len(models)
        all_overlap = []
        for idx_model_1 in range(n - 1):
            for idx_model_2 in range(idx_model_1 + 1, n):
                wcorr, _ = pearsonr(models[idx_model_1].coef_[0, :], models[idx_model_2].coef_[0, :])
                all_overlap.append(wcorr)
        #
        score = 2 * np.sum(all_overlap) / (n * (n - 1))
    elif method == "lasso":
        n = len(models)
        all_overlap = []
        for idx_model_1 in range(n - 1):
            for idx_model_2 in range(idx_model_1 + 1, n):
                overlap = get_pairwise_overlap(models[idx_model_1].coef_, models[idx_model_2].coef_)
                all_overlap.append(overlap)
        #
        score = 2 * np.sum(all_overlap) / (n * (n - 1))
    elif (method == "ridge"):
        n = len(models)
        all_overlap = []
        for idx_model_1 in range(n - 1):
            for idx_model_2 in range(idx_model_1 + 1, n):
                wcorr, _ = pearsonr(models[idx_model_1].coef_, models[idx_model_2].coef_)
                all_overlap.append(wcorr)
        #
        score = 2 * np.sum(all_overlap) / (n * (n - 1))

    return score


# multiview X
def split_train_test(X, Y, selected_idx):
    if str(type(X)) == "<class 'numpy.ndarray'>":
        X_train = X[selected_idx]
        Y_train = Y[selected_idx]
        # testing data
        X_test = X[np.logical_not(selected_idx)]
        Y_test = Y[np.logical_not(selected_idx)]
    else:
        X_train = {}
        X_test = {}
        K = len(X)
        for k in range(K):
            X_train[k] = X[k][selected_idx, :]
            X_test[k] = X[k][np.logical_not(selected_idx), :]

        Y_train = Y[selected_idx, :]
        Y_test = Y[np.logical_not(selected_idx), :]

    return (X_train, Y_train, X_test, Y_test)


# stabilility and generalizability
# X: matrix of explanatory variables (can be a list of matrices)
# Y: response matrix
# by now only can be applied to ddsPLS
# stability: True to obtain stability and generability scores, False to obtain generability score only

def parameter_optimization_ddspls(X, Y, stability=True, n_lamb=10, nsplits=10):
    all_lamb = np.linspace(0, get_max_lambda(X, Y), n_lamb)
    all_R = [r for r in range(1, Y.shape[1])]
    # get all splittings
    all_splits = None
    n_split = int(Y.shape[0] * 0.2)
    for _ in range(nsplits):
        # divide in train (80%) and testing (20%)
        # selected_idx = random.choices([False,True], weights=(20,80), k=X.shape[0])
        selected_idx = [True for _ in range(Y.shape[0] - n_split)] + [False for _ in range(n_split)]
        selected_idx = np.random.permutation(selected_idx)
        if all_splits is None:
            all_splits = selected_idx
        else:
            all_splits = np.vstack([all_splits, selected_idx])
    # best parameters found
    best_dist = 0
    best_possible_score = None
    if stability:
        best_possible_score = np.array([1, 1])
    else:
        best_possible_score = np.array([1])
    best_score = None
    best_R = None
    best_lambda = None
    # for each parametrization value calculate generalizability and stability
    for idx_lamb in range(len(all_lamb)):
        for idx_R in range(len(all_R)):
            all_models = []  # to calculate the model stability
            all_corr = []
            for idx_split in range(nsplits):
                selected_idx = all_splits[idx_split]
                # train data
                X_train, Y_train, X_test, Y_test = split_train_test(X, Y, selected_idx)
                # create model
                opt_model = py_ddspls.model.ddspls(X_train, Y_train, lambd=all_lamb[idx_lamb], R=all_R[idx_R], mode="reg", verbose=False)
                # save data to obtain generazability and stability
                hcorr = get_correlation(X_test, Y_test, opt_model.model)
                if not np.isnan(hcorr):
                    all_corr.append(hcorr)
                    all_models.append(opt_model.model)
            #
            if len(all_corr) == 0:
                continue
                # calculate generalizability
            gen_score = np.mean(np.abs(all_corr))
            # calculate stability
            new_score = None
            if stability:
                stab_score = get_stability(all_models)
                new_score = np.array([gen_score, stab_score])
            else:
                new_score = np.array([gen_score])

            new_dist = np.sum(np.square(new_score - best_possible_score))
            if best_score is None:
                best_score = new_score
                best_dist = new_dist
                best_R = all_R[idx_R]
                best_lambda = all_lamb[idx_lamb]
            elif new_dist < best_dist:
                best_score = new_score
                best_dist = new_dist
                best_R = all_R[idx_R]
                best_lambda = all_lamb[idx_lamb]

    return ({"R": best_R, "lambda": best_lambda, "score": best_score})


from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# stabilility and generalizability
# X: matrix of explanatory variables (can be a list of matrices)
# Y: response matrix
# by now only can be applied to SVR
# stability: True to obtain stability and generability scores, False to obtain generability score only
def parameter_optimization_svr(X, Y, stability=True, nsplits=10):
    all_eps = 10 ** np.linspace(-2, 0, 50)  # [0.1,0.2,0.4,0.8,1.0]
    all_C = [0.05, 0.1, 0.5, 1, 5, 10, 20, 40, 80, 100]
    # get all splittings
    all_splits = None
    n_split = int(Y.shape[0] * 0.2)
    for _ in range(nsplits):
        # divide in train (80%) and testing (20%)
        # selected_idx = random.choices([False,True], weights=(20,80), k=X.shape[0])
        selected_idx = [True for _ in range(Y.shape[0] - n_split)] + [False for _ in range(n_split)]
        selected_idx = np.random.permutation(selected_idx)
        if all_splits is None:
            all_splits = selected_idx
        else:
            all_splits = np.vstack([all_splits, selected_idx])
    # best parameters found
    best_dist = 0
    best_possible_score = None
    if stability:
        best_possible_score = np.array([1, 1])
    else:
        best_possible_score = np.array([1])
    best_score = None
    best_R = None
    best_lambda = None
    # for each parametrization value calculate generalizability and stability
    for idx_eps in range(len(all_eps)):
        for idx_C in range(len(all_C)):
            all_models = []  # to calculate the model stability
            all_qval = []
            for idx_split in range(nsplits):
                selected_idx = all_splits[idx_split]
                # train data
                X_train, Y_train, X_test, Y_test = split_train_test(X, Y, selected_idx)
                # create model
                model = svm.SVR(kernel="linear", C=all_C[idx_C], epsilon=all_eps[idx_eps])
                model.fit(X_train, Y_train)
                # save data to obtain generazability and stability
                qval = get_q_value(Y_test, model.predict(X_test), np.mean(Y_train))
                if (not np.isnan(qval)):  # not np.isnan(hcorr):
                    all_qval.append(qval)
                    all_models.append(model)
            #
            if len(all_qval) == 0:
                continue
                # calculate generalizability
            gen_score = np.mean(all_qval)
            # calculate stability
            new_score = None
            if stability:
                stab_score = get_stability(all_models, method="svr")
                new_score = np.array([gen_score, stab_score])
            else:
                new_score = np.array([gen_score])

            new_dist = np.sum(np.square(new_score - best_possible_score))
            if best_score is None:
                best_score = new_score
                best_dist = new_dist
                best_R = all_C[idx_C]
                best_lambda = all_eps[idx_eps]
            elif new_dist < best_dist:
                best_score = new_score
                best_dist = new_dist
                best_R = all_C[idx_C]
                best_lambda = all_eps[idx_eps]

    return ({"C": best_R, "epsilon": best_lambda, "score": best_score})


from sklearn.linear_model import Lasso


# stabilility and generalizability
# X: matrix of explanatory variables (can be a list of matrices)
# Y: response matrix
# by now only can be applied to Lasso
# stability: True to obtain stability and generability scores, False to obtain generability score only
def parameter_optimization_lasso(X, Y, stability=True, nsplits=10):
    all_alpha = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5, 10, 50, 100, 500]  # 10**np.linspace(3,-4,100)*0.5
    # get all splittings
    all_splits = None
    n_split = int(Y.shape[0] * 0.2)
    for _ in range(nsplits):
        # divide in train (80%) and testing (20%)
        # selected_idx = random.choices([False,True], weights=(20,80), k=X.shape[0])
        selected_idx = [True for _ in range(Y.shape[0] - n_split)] + [False for _ in range(n_split)]
        selected_idx = np.random.permutation(selected_idx)
        if all_splits is None:
            all_splits = selected_idx
        else:
            all_splits = np.vstack([all_splits, selected_idx])
    # best parameters found
    best_dist = 0
    best_possible_score = None
    if stability:
        best_possible_score = np.array([1, 1])
    else:
        best_possible_score = np.array([1])
    best_score = None
    best_alpha = None
    # for each parametrization value calculate generalizability and stability
    for idx_alpha in range(len(all_alpha)):
        all_models = []  # to calculate the model stability
        all_qval = []
        for idx_split in range(nsplits):
            selected_idx = all_splits[idx_split]
            # train data
            X_train, Y_train, X_test, Y_test = split_train_test(X, Y, selected_idx)
            # create model
            model = Lasso(alpha=all_alpha[idx_alpha], max_iter=200000)
            model.fit(X_train, Y_train)
            # save data to obtain generazability and stability
            qval = get_q_value(Y_test, model.predict(X_test), np.mean(Y_train))
            # print(str(rval) + " " + str(qval))
            if (not np.isnan(qval)):
                all_qval.append(qval)
                all_models.append(model)
        #
        if len(all_qval) == 0:
            continue
            # calculate generalizability
        gen_score = np.mean(all_qval)
        # calculate stability
        new_score = None
        if stability:
            stab_score = get_stability(all_models, method="lasso")
            new_score = np.array([gen_score, stab_score])
        else:
            new_score = np.array([gen_score])
        new_dist = np.sum(np.square(new_score - best_possible_score))
        if best_score is None:
            best_score = new_score
            best_dist = new_dist
            best_alpha = all_alpha[idx_alpha]
        elif new_dist < best_dist:
            best_score = new_score
            best_dist = new_dist
            best_alpha = all_alpha[idx_alpha]

    return ({"alpha": best_alpha, "score": best_score})


# DDSPLS
# H0: “There is no relationship between the response variable and the multiple views,
# theref/ore the correlation obtained with the original data is not different from the correlation obtained with the permuted data

# ML
# H0: “It is not possible to predict the response variables given the explanatory variables,
# therefore the Q² obtained with the original data is not different from the Q² obtained with the permuted data

# stability: True to obtain stability and generability scores, False to obtain generability score only
def get_pvalue(X_opt, Y_opt, X_hold, Y_hold, hcorr, parameters, method="ddspls", B=2000):
    all_corr = []
    if method == "ddspls":
        for _ in range(B):
            pY_opt = np.random.permutation(Y_opt)
            opt_model = py_ddspls.model.ddspls(X_opt, pY_opt, lambd=parameters["lambda"], R=parameters["R"], mode="reg", verbose=False)
            model = opt_model.model
            # obtain correlation
            ncorr = get_correlation(X_hold, Y_hold, model)
            if not np.isnan(ncorr):
                all_corr.append(np.abs(ncorr))
    elif method == "svr":
        for _ in range(B):
            pY_opt = np.random.permutation(Y_opt)
            model = svm.SVR(kernel="linear", C=parameters["C"], epsilon=parameters["epsilon"])
            model.fit(X_opt, pY_opt)
            # obtain correlation
            # ncorr = get_correlation(X_hold,Y_hold,model,method = "svr")
            ncorr = get_q_value(Y_hold, model.predict(X_hold), np.mean(Y_opt))
            if not np.isnan(ncorr):
                all_corr.append(ncorr)
    elif method == "lasso":
        for _ in range(B):
            pY_opt = np.random.permutation(Y_opt)
            model = Lasso(alpha=parameters["alpha"], max_iter=200000)
            model.fit(X_opt, pY_opt)
            # obtain correlation
            # ncorr = get_correlation(X_hold,Y_hold,model,method = "lasso")
            ncorr = get_q_value(Y_hold, model.predict(X_hold), np.mean(Y_opt))
            if not np.isnan(ncorr):
                all_corr.append(ncorr)
    elif method == "ridge":
        for _ in range(B):
            pY_opt = np.random.permutation(Y_opt)
            model = Ridge(alpha=parameters["alpha"])
            model.fit(X_opt, pY_opt)
            # obtain correlation
            # ncorr = get_correlation(X_hold,Y_hold,model,method = "ridge")
            ncorr = get_q_value(Y_hold, model.predict(X_hold), np.mean(Y_opt))
            if not np.isnan(ncorr):
                all_corr.append(ncorr)

    #
    # print(all_corr)
    pval = np.sum(np.array(all_corr) >= hcorr) / (len(all_corr) + 1)
    #
    return pval


from statsmodels.stats.multitest import fdrcorrection


# method: ddspls/SVR (linear)
# ddspls: Y can be a n*k matrix
# svr: Y is a n*1 matrix

# stability: True to obtain stability and generability scores, False to obtain generability score only
def holdout_framework(X, Y, method="ddspls", n_hold=10, alpha=0.05, stability=True):
    all_corr = []
    all_pval = []
    all_models = []
    n_split = int(Y.shape[0] * 0.2)
    for idx_hold in range(n_hold):
        selected_idx = [True for _ in range(Y.shape[0] - n_split)] + [False for _ in range(n_split)]
        selected_idx = np.random.permutation(selected_idx)
        X_opt, Y_opt, X_hold, Y_hold = split_train_test(X, Y, selected_idx)
        # normalize data
        if method != "ddspls":
            X_scaler = StandardScaler()
            X_opt = X_scaler.fit_transform(X_opt)
            #
            X_hold = X_scaler.transform(X_hold)
        # get best parameters
        print("parameter optimization holdout set #" + str(idx_hold))
        parameters = None
        if method == "ddspls":
            parameters = parameter_optimization_ddspls(X_opt, Y_opt, stability=stability, nsplits=50)
        elif method == "svr":
            parameters = parameter_optimization_svr(X_opt, Y_opt, stability=stability)
        elif method == "lasso":
            parameters = parameter_optimization_lasso(X_opt, Y_opt, stability=stability)
        elif method == "ridge":
            parameters = parameter_optimization_ridge(X_opt, Y_opt, stability=stability)
        # train all optimization set with the found parameter
        print("fitting to optimization set #" + str(idx_hold))
        model = None
        if method == "ddspls":
            opt_model = py_ddspls.model.ddspls(X_opt, Y_opt, lambd=parameters["lambda"], R=parameters["R"], mode="reg", verbose=False)
            model = opt_model.model
        elif method == "svr":
            model = svm.SVR(kernel="linear", C=parameters["C"], epsilon=parameters["epsilon"])
            model.fit(X_opt, Y_opt)
        elif method == "lasso":
            model = Lasso(alpha=parameters["alpha"], max_iter=200000)
            model.fit(X_opt, Y_opt)
        elif method == "ridge":
            model = Ridge(alpha=parameters["alpha"])
            model.fit(X_opt, Y_opt)
        # get correlation in holdout set

        print("calculating correlation and pvalue of holdout set #" + str(idx_hold))
        hcorr = None
        if method == "ddspls":
            hcorr = get_correlation(X_hold, Y_hold, model, method=method)
        else:
            hcorr = get_q_value(Y_hold, model.predict(X_hold), np.mean(Y_opt))
        # obtain pval for the correlation
        if method == "ddspls":
            pval = get_pvalue(X_opt, Y_opt, X_hold, Y_hold, np.abs(hcorr), parameters, method=method)
        else:
            pval = get_pvalue(X_opt, Y_opt, X_hold, Y_hold, hcorr, parameters, method=method)
        # save all
        all_corr.append(hcorr)
        all_pval.append(pval)
        all_models.append(model)
        print(str(hcorr) + " " + str(pval))
    # fdr/bonferroni correction and return the best model
    print("correcting pvalues")
    corrected_pval = fdrcorrection(all_pval, alpha=alpha)
    print(corrected_pval)
    pos = [i for i, x in enumerate(list(corrected_pval[0])) if x]
    # check if a significant p-value was found
    if (len(pos) != 0):
        # significant pvalues were found
        # more than one region found
        # for each region obtain the pvalues for each variable (then apply FDR)
        best_pos = None
        for idx_pval in pos:
            if best_pos is None:
                best_pos = idx_pval
            else:
                if (corrected_pval[1][idx_pval] < corrected_pval[1][best_pos]):
                    best_pos = idx_pval
                elif (corrected_pval[1][idx_pval] == corrected_pval[1][best_pos]):
                    if all_corr[idx_pval] > all_corr[best_pos]:
                        best_pos = idx_pval

        return ({"model": all_models[best_pos], "corr": all_corr[best_pos], "pval": corrected_pval[1][best_pos]})
    else:
        return None


# method to vertically stack matrix B
def vstack_matrix(A, B):
    R = None
    if A is None:
        R = B
    else:
        R = np.vstack([A, B])
    return R
