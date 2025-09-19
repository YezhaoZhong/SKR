#!/usr/bin/env python

# import modules

import pandas as pd
import networkx as nx
from networkx.algorithms.bipartite.matrix import biadjacency_matrix
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import random
from sklearn import metrics
from joblib import Parallel, delayed
from sklearn.utils import parallel_backend
import warnings
from Models import runModels, setDist
from collections import defaultdict
from scipy.spatial.distance import cdist

random.seed(1949)
np.random.seed(1949)

def option(str1, str2, str3):
    """
    Define the options as global variables.

    Parameters
    ----------
    str1: string
        String for choosing a prediction method.
    str2: string
        String for choosing a tuning metric.
    str3: string
        String for choosing CV or Nested CV.

    Returns
    -------
    None
    """
    global methodOption
    global tuningMetrice
    global validationOption

    methodOption = str1
    tuningMetrice = str2
    validationOption = str3


def Pairs2Mat(path,colname1,colname2,sep = "\t"):
    """
    Transfer pairs into matrix form.

    Parameters
    ----------
    path: string
        Path of the data.
    colname1: string
        Column name of the first value of the pairs.
    colname2: string
        Column name of the second value of the pairs.

    Returns
    -------
    matrix_all:
        Drug*feature or drug*ADR dataframe
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    df = pd.read_csv(path,sep=sep,low_memory=False)
    df_nadroped = df[[colname1,colname2]].dropna()
    drug_names = df_nadroped[colname1] # put col of df in var
    drug_names_upper = [drug_name.upper() for drug_name in drug_names]
    features = df_nadroped[colname2]
    features_upper = [feature.upper() for feature in features]

    edgelist = zip(features_upper, drug_names_upper)
    ###making Biparite Graph###
    B = nx.DiGraph()
    B.add_nodes_from(features_upper,bipartite = 0)
    B.add_nodes_from(drug_names_upper,bipartite = 1)
    B.add_edges_from(edgelist)
    drug_nodes = {n for n, d in B.nodes(data=True) if d['bipartite']==1}
    feature_nodes = {n for n, d in B.nodes(data=True) if d['bipartite']==0}
    drug_nodes = list(drug_nodes)
    drug_nodes.sort()
    feature_nodes = list(feature_nodes)
    feature_nodes.sort()
    ###Getting the Bi-Adjacency matrix between adr and drugs ###
    matrix_all = biadjacency_matrix(B, row_order = feature_nodes, column_order = drug_nodes) # create biadjacency matrix for drug adr graph
    matrix_all = pd.DataFrame(matrix_all.A.T, columns=feature_nodes, index=drug_nodes)
    
    return matrix_all # drug*feature or drug*adr

def FeaturePreprocess(df_all, drug):
    """
    Find the intersection drugs of ADR data and feature data.

    Parameters
    ----------
    df_all: dataframe
        Feature data matrix
    drug: list
        drug list in the 

    Returns
    -------
    featrueMat
        Array of drug*feature or drug*ADR 
    """
    drug_nodes_df = np.intersect1d(df_all.index, drug)
    df = df_all.loc[drug_nodes_df]
    _, q = df.shape
    drug_nodes_diff = np.setdiff1d(drug, (df.index).tolist())
    n = len(drug_nodes_diff)
    df_diff = pd.DataFrame(np.zeros((n, q)))
    df_diff.index = drug_nodes_diff
    df_diff.columns = df.columns
    df_all = pd.concat([df, df_diff], axis = 0)
    featureMat = df_all.loc[drug]
    return np.array(featureMat)


def fold(idx_train,idx_test,feature_matrix,matrix,par):
    """
    Make prediction for the validation set in a fold of CV or in the test set.

    Parameters
    ----------
    idx_train: array
        Rows idx chosen as training set.
    idx_test: array
        Rows idx chosen as test set.
    feature_matrix: array
        Feature matrix.
    matrix: array
        ADR matrix
    par: list
        List of hyperparameters.

    Returns
    -------
    results: dict
        Dictionary includes the evaluation metrics
    """
    print(methodOption + ' starts:')
    X = np.array(feature_matrix[idx_train, :].copy())
    X_new = np.array(feature_matrix[idx_test, :].copy())
    Y = np.array(matrix[idx_train, :].copy())
    y_gt = np.array(matrix[idx_test, :].copy())
    y_new = runModels(Y=Y,X=X,X_new=X_new,method_option=methodOption,par=par)
    print(methodOption + ' ends:')

    results = {}

    # -----perdrug -----"

    ALL_aupr = np.zeros(len(idx_test))
    ALL_auroc = np.zeros(len(idx_test))
    ALL_F1score = np.zeros(len(idx_test))
    for i in range(len(idx_test)):
        if y_gt[i, :].sum() == 0:
            ALL_auroc[i] = np.nan
            ALL_aupr[i] = np.nan
        elif y_new[i, :].sum() == 0:
            ALL_auroc[i] = 0
            ALL_aupr[i] = np.nan
        else:
            fpr, tpr, threshold = metrics.roc_curve(y_gt[i, :], y_new[i, :])
            ALL_auroc[i] = auc(fpr, tpr)
            prec, recall, threshold = precision_recall_curve(y_gt[i, :], y_new[i, :])
            ALL_aupr[i] = auc(recall, prec)

    results["AUPRperdrug"] = np.nanmean(ALL_aupr)
    results["AUROCperdrug"] = np.nanmean(ALL_auroc)
    results["AUPR+AUROCperdrug"] = results["AUPRperdrug"] + results["AUROCperdrug"]

    prec, recall, prthreshold = precision_recall_curve(y_gt.ravel(), y_new.ravel())
    results["AUPR"] = auc(recall, prec)

    fpr, tpr, rocthreshold = metrics.roc_curve(y_gt.ravel(), y_new.ravel())
    results["AUROC"] = auc(fpr, tpr)

    results["AUPR+AUROC"] = results["AUPR"] + results["AUROC"]

    print("-----------")
    keys = results.keys()
    for key in keys:
        print(f"{key}: {results[key]}")
    print("-----------")

    return results

def innerfold(idx_train,idx_test,feature_matrix,matrix,par):
    """
    Make prediction to tune hyperparameters for the validation set in Nested CV innerfolds or in CV folds.

    Parameters
    ----------
    idx_train: array
        Rows idx chosen as training set.
    idx_test: array
        Rows idx chosen as test set.
    feature_matrix: array
        Feature matrix.
    matrix: array
        ADR matrix.
    par: list
        List of hyperparameters.

    Returns
    -------
    result: dict
        Dictionary includes the evaluation metrics
    """
    X = np.array(feature_matrix[idx_train, :].copy())
    X_new = np.array(feature_matrix[idx_test, :].copy())
    Y = np.array(matrix[idx_train, :].copy())
    y_gt = np.array(matrix[idx_test, :].copy())
    y_new = runModels(Y=Y,X=X,X_new=X_new,method_option=methodOption,par=par)


    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if tuningMetrice == "AUROCperdrug": 
        ALL_auroc = np.zeros(len(idx_test))
        for i in range(len(idx_test)):
            if y_gt[i, :].sum() == 0:
                ALL_auroc[i] = np.nan
            elif y_new[i, :].sum() == 0:
                ALL_auroc[i] = 0
            else:
                fpr, tpr, threshold = metrics.roc_curve(y_gt[i, :], y_new[i, :])
                ALL_auroc[i] = auc(fpr, tpr) 
        result = np.nanmean(ALL_auroc)
    elif tuningMetrice == "AUPRperdrug": 
        ALL_aupr = np.zeros(len(idx_test))
        for i in range(len(idx_test)):
            if y_gt[i, :].sum() == 0:
                ALL_aupr[i] = np.nan
            elif y_new[i, :].sum() == 0:
                ALL_aupr[i] = np.nan
            else:
                prec, recall, threshold = precision_recall_curve(y_gt[i, :], y_new[i, :])
                ALL_aupr[i] = auc(recall, prec)
        result = np.nanmean(ALL_aupr)
    elif tuningMetrice == "AUROC": 
        fpr, tpr, rocthreshold = metrics.roc_curve(y_gt.ravel(), y_new.ravel())
        result = auc(fpr, tpr)
    elif tuningMetrice == "AUPR": 
        if y_new.ravel().sum() == 0:
            result = 0
        else: 
            prec, recall, threshold = precision_recall_curve(y_gt.ravel(), y_new.ravel())
            result = auc(recall, prec)
    else:
    
        raise ValueError("Please select a metrice for tuning. Choose AUPRperdrug, AUROCperdrug, AUPR or AUROC.")
    return result


def setvar_tune(size, f):
    """
    Set a global variable to store the results of folds (CV) or innerfolds (Nested CV) for the hyperparameter set.

    Parameters
    ----------
    size: int
        The number of hyperparameter combinations.
    f: int
        The number of fold.

    Returns
    -------
    result: dict
        Dictionary includes the evaluation metrics for each hyperparameter combination.
    """
    global ALL_tuning_metrices
    ALL_tuning_metrices = np.zeros((size, f))

def asgvar_tune(idx, f, results):
    """
    Assign the results of folds (CV) or innerfolds (Nested CV) for the hyperparameter set.

    Parameters
    ----------
    idx: int
        the index of a hyperparameter combination.
    f: int
        The index of innerfold of Nested CV.
    results: dict
        Dictionary includes the evaluation metrics for each hyperparameter combination.

    Returns
    -------
    None
    """
    ALL_tuning_metrices[idx, f] = results

def tuning_results(tuneVar):
    """
    Find the best combination of hyperparameters within the results of folds (CV) or innerfolds (Nested CV).

    Parameters
    ----------
    tuneVar: list
        List of hyperparameter combinations.

    Returns
    -------
    Var: float
       The best combination of hyperparameters.
    Value: float
       The metric of tuned hyperparameters.
    """
    mean = ALL_tuning_metrices.mean(axis=1)
    idx = np.argmax(mean)
    Var = tuneVar[idx]
    Value = mean[idx]
    print(f"best hyperpar: {Var}")
    print(f"{tuningMetrice}: {Value}")
    print(f"{tuningMetrice} for each fold: {ALL_tuning_metrices[idx, :]}")

    return Var, Value


def setvar_cv():
    """
    Define a global variable to store validation results of Nested CV outerfold. 

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    global metrices
    metrices = defaultdict(list)

def asgvar_cv(results):
    """
    Assgin validation results of Nested CV outerfold. 

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    keys = results.keys()
    for key in keys:
        metrices[key].append(results[key])


def cv_results():
    """
    Print and return validation results of Nested CV. 

    Parameters
    ----------
    None

    Returns
    -------
    metrices: dict
        Dictionary that includes results of all metrics for each fold.
    """
    keys = metrices.keys()
    for key in keys:
        print(f"Mean {key}: {(np.array(metrices[key])).mean()}, std: {(np.array(metrices[key])).std()}")
    print("-----------")
    return dict(metrices)


def tuning_loop(innermatrix, idx_train_inner, idx_test_inner, feature_matrix_inner, hyperparList, i, f):
    """
    Define a tuning loop function for parallel computation.

    Parameters
    ----------
    innermatrix: array
        ADR matrix.
    idx_train_inner: array
        Index of training set.
    idx_test_inner: array
        Index of validation set.
    feature_matrix_inner: array
        Drug-feature matrix.
    hyperparList: list
        List of hyperparameter combinations.
    i: int
        the index of a hyperparameter combination.
    f: int
        The index of innerfold of Nested CV.

    Returns
    -------
    None
    """
    # print('test set size:', len(idx_test_inner))
    results = innerfold(idx_train_inner,idx_test_inner,feature_matrix=feature_matrix_inner,matrix=innermatrix,par=hyperparList[i])
    asgvar_tune(i, f, results=results)
    # print("------ lmd1: ", l1, "lmd1: ", l2, "sigma: ", s, "------")


def evaluation(Y, X, method_option, tuning_metrice, hyperparList=[0], hyperparfixed=False, Validation=False,  n_jobs=10):
    """
    Main function. This function allows two evaluation process, Cross Validation (CV) and Nested CV. Y was first split into training set and test set. Then training set was used for CV or Nested CV. As Nested CV already includes the hyperparameter tuning process, the results would be reliable and no need to use test set; while CV requires to use test set to validate the methods with tuned hyperparameters in training set. 

    Parameters
    ----------
    Y: array
        ADR matrix.
    X: array
        Feature matrix.
    method_option: str
        Prediction method. (VKR, SKR, KRR, Naive, LNSM_RLN", LNSM_jaccard, RF, BRF, SVM...)
    tuning_metrice: str
        Tune metric for CV or Nested CV. (AUPR, AUROC, AUPRperdrug, AUROCperdrug)
    hyperparList: list
        List of hyperparameter combinations.
    hyperparfixed: bool
        Skip the tuning step or not. If the input of hyperparList is a set of hyperparameters that needs to be tuned then hyperparfixed=False, otherwise, the input of hyperparList is tuned hyperparafixed=True.
    Validation: str
        Choose CV or Nested CV (cv, nested_cv).
    n_jobs: int
        Number of parallel jobs.

    Returns
    -------
    If Validation = 'nested cv'
        out: dict
            The results of all metrics for each outer fold.
        bestHyperParsOut:
            The tuned hyperparameters for each outer fold.

    If Validation = 'cv'
        results: dict
            The results of test set.
        bestHyperPars: 
            The tuned hyperparameters for test set.
    """
    random.seed(1949)
    np.random.seed(1949)
    option(method_option, tuning_metrice, Validation)

    matrix_all = np.array(Y.copy()).astype(float)
    m_all,n_all = matrix_all.shape # number of drug # number of side effect 
    drug = Y.index
    
    
    ### Setting validation set / training set / testing set ###
    validate_sz = int(0.25 * m_all)
    IDX_all = list(range(m_all))
    random.shuffle(IDX_all)
    IDX_validate = sorted(IDX_all[0:validate_sz])
    # print("first few validation set idx:")
    # print(IDX_validate[0:10])
    IDX_validate_diff = np.setdiff1d(IDX_all, IDX_validate)
    matrix = matrix_all[IDX_validate_diff, :].copy().astype(float)

    df = X
    featureMat_all = FeaturePreprocess(df, drug=drug).astype(float)
    featureMat = featureMat_all[IDX_validate_diff, :].copy().astype(float)
    
    
    # non_zero_idx_union = np.hstack(np.where(~((featureMat.sum(1) == 0) & (featureMat.sum(1) == 0))))
    # non_zero_idx_missing = np.hstack(np.where(~(~(featureMat.sum(1) == 0) & ~(featureMat.sum(1) == 0))))
    non_zero_idx_intersect = np.hstack(np.where(~(featureMat.sum(1) == 0) & ~(matrix.sum(1) == 0)))

    # intersect
    non_zero_idx_intersect_all = np.hstack(np.where(~(featureMat_all.sum(1) == 0) & ~(matrix_all.sum(1) == 0)))

    matrix_all = matrix_all[non_zero_idx_intersect_all, :].copy()
    featureMat_all = featureMat_all[non_zero_idx_intersect_all, :].copy()

    matrix = matrix[non_zero_idx_intersect, :].copy()
    featureMat = featureMat[non_zero_idx_intersect, :].copy()

    IDX_validate = np.setdiff1d(non_zero_idx_intersect_all, IDX_validate_diff)
    IDX_validate_diff = np.setdiff1d(non_zero_idx_intersect_all, IDX_validate)


    distance_X = cdist(featureMat, featureMat)**2
    distance_Y = cdist(matrix, matrix)**2

    distance_X_all = cdist(featureMat_all, featureMat_all)**2
    distance_Y_all = cdist(matrix_all, matrix_all)**2


    drug_nodes_intersect_all = np.array(drug)[non_zero_idx_intersect_all]
    drug_nodes_intersect_validate_diff = np.array(drug)[IDX_validate_diff]
    drug_nodes_intersect_validate = np.array(drug)[IDX_validate]

    IDX_validate = np.array([x for x in range(len(drug_nodes_intersect_all)) if drug_nodes_intersect_all[x] in drug_nodes_intersect_validate])
    IDX_validate_diff = np.array([x for x in range(len(drug_nodes_intersect_all)) if drug_nodes_intersect_all[x] in drug_nodes_intersect_validate_diff])
    

    m,n = matrix.shape  # number of drug # number of side effect


    FOLDS = 5
    innerFOLDS = 4
    ####for test sets####
    setvar_cv()

    sz = m
    IDX = list(range(sz))
    fsz = int(sz/FOLDS)
    random.shuffle(IDX)
    IDX = np.array(IDX)
    offset = 0

    innersz = sz - fsz
    innerIDX = list(range(innersz))
    random.shuffle(innerIDX)
    innerIDX = np.array(innerIDX)
    innerfsz = int(innersz / innerFOLDS)
    inneroffset = 0

    if Validation == "nested_cv":
        bestHyperParsOut=[]
        print("---------- nested cv start ----------")
        for f in range(FOLDS):
            offset = 0 + fsz*f 
            idx_test = IDX[offset:offset + fsz]
    
            idx_train = IDX[np.setdiff1d(np.arange(len(IDX)), np.arange(offset,offset + fsz))]
            print("Fold:",f)
                
            innermatrix = matrix[idx_train, :].copy()
            innerfeatureMat = featureMat[idx_train, :].copy()
            innerdistance_X = distance_X[np.ix_(idx_train, idx_train)].copy()
            innerdistance_Y = distance_Y[np.ix_(idx_train, idx_train)].copy()
    
            # setvar_besttune(innerFOLDS)
    
            setvar_tune(len(hyperparList), f = innerFOLDS)
            print("number of hyperpars combination: ", len(hyperparList))
            print("first few training idx: ", np.sort(idx_train[0:10]))
            print("first few testing idx: ", np.sort(idx_test[0:10]))
            if hyperparfixed == False:
                for innerf in range(innerFOLDS):
                    inneroffset = 0 + innerf*innerfsz
                    idx_test_inner = innerIDX[inneroffset:inneroffset + innerfsz]
                    idx_train_inner = innerIDX[np.array(np.setdiff1d(np.arange(len(idx_train)), np.arange(inneroffset,  inneroffset + innerfsz)))]

                    setDist(dist_X=innerdistance_X[np.ix_(idx_train_inner, idx_train_inner)], dist_Y=innerdistance_Y[np.ix_(idx_train_inner, idx_train_inner)], dist_X_new=innerdistance_X[np.ix_(idx_test_inner, idx_train_inner)])
                    # setDist()
    
                    print("Inner Fold:", innerf)
                    with parallel_backend('threading'):
                        Parallel(n_jobs=n_jobs)(delayed(tuning_loop)(innermatrix = innermatrix, idx_train_inner = idx_train_inner, idx_test_inner = idx_test_inner, feature_matrix_inner = innerfeatureMat, hyperparList = hyperparList, i = i, f=innerf) for i in range(len(hyperparList)))
                    setDist()
                bestHyperPars, evalValue = tuning_results(tuneVar=hyperparList)
                bestHyperParsOut.append(bestHyperPars)
            else:
                setDist()
                bestHyperPars = hyperparfixed[f]
                bestHyperParsOut.append(bestHyperPars)
            
    
            # asg_besttune(innerf, value=evalValue, var=hyperpars)
                # raise ValueError("--.")
                    
            # _, bestHyperPars = besttune()
    
            print("--- tuning end ---")
            print('target size:', len(idx_test))
            print("------ best hyper pars: ", bestHyperPars, "------")
            results = fold(idx_train,idx_test,featureMat,matrix,par=bestHyperPars)
            asgvar_cv(results=results)

        out = cv_results()
        return out, bestHyperParsOut

    elif Validation == "cv":
        print("---------- cv start ----------")

        setvar_tune(len(hyperparList), f = FOLDS)
        if hyperparfixed == False:
            
            for f in range(FOLDS):
                offset = 0 + fsz*f 
                idx_test = IDX[offset:offset + fsz]
                idx_train = IDX[np.setdiff1d(np.arange(len(IDX)), np.arange(offset,offset + fsz))]
                setDist(dist_X=distance_X_all[np.ix_(idx_train, idx_train)], dist_Y=distance_Y_all[np.ix_(idx_train, idx_train)], dist_X_new=distance_X_all[np.ix_(idx_test, idx_train)])
                # setDist()

                print("Fold:", f)

                with parallel_backend('threading'):
                    Parallel(n_jobs=n_jobs)(delayed(tuning_loop)(innermatrix = matrix, idx_train_inner = idx_train, idx_test_inner = idx_test, feature_matrix_inner = featureMat, hyperparList = hyperparList, i = i, f=f)for i in range(len(hyperparList)))
                setDist()
            bestHyperPars, evalValue = tuning_results(tuneVar=hyperparList)

        else:
            setDist()
            bestHyperPars = hyperparfixed

    
        print("--- tuning end ---")
        idx_test = IDX_validate
        idx_train = IDX_validate_diff
        print('target size:', len(idx_test))
        print("------ best hyper pars: ", bestHyperPars, "------")
        results = fold(idx_train,idx_test,featureMat_all,matrix_all,par=bestHyperPars)
        return results, bestHyperPars


