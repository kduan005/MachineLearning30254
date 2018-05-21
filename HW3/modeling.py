from __future__ import division
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, \
    OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_curve
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
import sys
#reference:https://github.com/rayidghani/magicloops/blob/master/magicloop.py

def classifiers_parameters(grid_type = "standard"):
    '''
    To generate a dictionary of classifiers and a dictionary containing
    corresponding parameters to each classifier
    Inputs:
    grid_type: strings, has two values of "standard" and "test". If standard,
    provide parameters of different level for selecting best performing model;
    if "test", provide default parameters
    '''

    classifiers = {"RF": RandomForestClassifier(n_estimators = 50, n_jobs = -1),\
        "LR": LogisticRegression(penalty = "l1", C = 1e5),\
        "SVM": svm.SVC(kernel = "linear", probability = True, random_state = 0),\
        "DT": DecisionTreeClassifier(),\
        "KNN": KNeighborsClassifier(n_neighbors = 3),\
        "NB": GaussianNB(),\
        "AB": AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), n_estimators = 200),\
        "BAG": BaggingClassifier(DecisionTreeClassifier(max_depth = 10), \
        n_estimators = 5, max_samples = 0.65, max_features = 1)}

    standard_grid = {
        "RF": {"n_estimators": [1, 10], "max_depth":[1, 5],\
            "max_features": ["sqrt", "log2"], "min_samples_split": [2, 20]},\
        "LR": {"penalty": ["l1", "l2"], "C": [10 ** i for i in range(-2, 2)]},\
        "SVM": {"C": [10 ** i for i in range(-2,2)], "kernel": ["linear"]},\
        "DT": {"criterion": ["gini", "entropy"], "max_depth": [1, 10],\
            "max_features":["sqrt", "log2"], "min_samples_split": [2, 10]},\
        "KNN": {"n_neighbors": [5, 10], "weights": ["uniform", \
            "distance"], "algorithm": ["auto", "ball_tree", "kd_tree"]},\
        "NB": {},
        "AB": {"algorithm": ["SAMME", "SAMME.R"], "n_estimators": \
            [10 ** i for i in range(0, 3)]},\
        "BAG": {"n_estimators": [5, 10, 20], "max_samples": [0.35, 0.5, 0.65]}
        }

    test_grid = {
        "RF":{"n_estimators": [1], "max_depth": [1], "max_features": ["sqrt"],\
            "min_samples_split": [10]},\
        "LR": { "penalty": ["l1"], "C": [0.01]},\
    	"SVM": {"C" :[0.01],"kernel":["linear"]},\
    	"DT": {"criterion": ["gini"], "max_depth": [1], "max_features": ["sqrt"],\
            "min_samples_split": [10]},\
        "KNN": {"n_neighbors": [5], "weights": ["uniform"], "algorithm": ["auto"]},\
        "NB": {},\
        "AB": {"algorithm": ["SAMME"], "n_estimators": [1]},\
        "BAG": {"n_estimators": [5]}
        }

    if grid_type == "test":
        param_grid = test_grid
    else:
        param_grid = standard_grid

    return classifiers, param_grid

def split_train_test(X, y, test_size = 0.25, temporal = False, temporal_var = None, \
split_date = None):
    '''
    temporal: boolean, indicator for whether to employ temporal validation or not
    temporal_var: string, name of the temporal feature
    split_date: a list of Timestamps, listing multiple Timestamps for spliting \
    training and testing sets
    '''
    X_y_sets = []
    if temporal is False:
        X = X.drop([temporal_var], axis = 1).copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = \
            test_size, random_state = 0)
        X_y_sets.append((X_train, X_test, y_train, y_test))

    else:

        for date in split_date:
            X_train = X[X["date_posted"] <= date].copy()
            X_train = X_train.drop([temporal_var], axis = 1).copy()
            X_test = X[X["date_posted"] > date].copy()
            X_test = X_test.drop([temporal_var], axis = 1).copy()
            y_train = y[X["date_posted"] <= date].copy()
            y_test = y[X["date_posted"] > date].copy()
            X_y_sets.append((X_train, X_test, y_train, y_test))
    return X_y_sets

def classifier_loop(models_of_interest, classifiers, param_grid, X_train, X_test,\
 y_train, y_test, output = True, loop_no = None):
    '''
    Inputs:
        models_of_interest:a list of strings, each string stands for a model of
        interest
        classifiers: dictionary of classifiers, return from classifiers_parameters
        param_grid: dictionary of parameters for different types of classifiers,
        return from classifiers_parameters
        X: pandas DataFrame, training set
        y: pandas DataFrame, testing set
        output: boolean, if output needed to be save to files or not
        temporal: boolean, indicator for temporal validation. If true, then user
        need to input n_split to indicate the number of splits
    '''

    results_df = pd.DataFrame(columns = ("model_type", "classifier", "parameters",\
        "train_time", "test_time", "accuracy", "f1_score", "precision", "recall",\
        "auc", "p_at_1", "p_at_2", "p_at_5", "p_at_10", "p_at_20", "p_at_30", \
        "p_at_50", "r_at_1", "r_at_2", "r_at_5", "r_at_10", "r_at_20", "r_at_30", \
        "r_at_50"))

    for index, classifier in enumerate([classifiers[x] for x in models_of_interest]):
        print (models_of_interest[index])
        parameter_values = param_grid[models_of_interest[index]]
        for p in ParameterGrid(parameter_values):
            try:
                classifier.set_params(**p)
                print (classifier)
                train_start = time.time()
                classifier.fit(X_train, y_train)
                train_end = time.time()
                train_time = train_end - train_start
                print("training done")

                test_start = time.time()
                y_pred = classifier.predict(X_test)
                test_end = time.time()
                test_time = test_end - test_start
                print("prediction done")

                y_pred_probs = classifier.predict_proba(X_test)[:, 1]
                print ("prediction_prob done")
                scores = evaluate_matrics(y_test, y_pred, y_pred_probs)
                print("evaluation done")

                row_index = len(results_df)
                model_name = models_of_interest[index] + str(row_index)
                
                results_df.loc[row_index] = [models_of_interest[index], classifier,\
                    p, train_time, test_time, scores["accuracy"], \
                    scores["f1_score"], scores["precision"], \
                    scores["recall"], scores["auc"], \
                    scores["p_at_1"], scores["p_at_2"], scores["p_at_5"],\
                    scores["p_at_10"], scores["p_at_20"], scores["p_at_30"], \
                    scores["p_at_50"], scores["r_at_1"], scores["r_at_2"], \
                    scores["r_at_5"], scores["r_at_10"], scores["r_at_20"], \
                    scores["r_at_30"], scores["r_at_50"]]
                if output:
                    plot_precision_recall(y_test, y_pred_probs, model_name, loop_no)
            except MemoryError as e:
                print ("Error:", e)
                continue

    if output:
        results_df.to_csv("evaluation/clf_evaluations_{}.csv".format(loop_no))

    return results_df

def evaluate_matrics(y_true, y_pred, y_pred_probs):
    '''
    To generate evaluation matrics for each model including:
    (y_true, y_pred):accuracy, f1 score, precision, recall, roc_auc_score
    (y_true, y_pred_probs): precision and recall at different levels of
    5%, 10%, 20% rate of intervention
    '''

    evaluation_results = {}

    evaluation_metrics = {"accuracy": accuracy_score, "f1_score": f1_score, \
    "precision": precision_score, "recall": recall_score, "auc": roc_auc_score}

    for metric, fn in evaluation_metrics.items():
        evaluation_results[metric] = fn(y_true, y_pred)

    y_pred_probs_sorted, y_true_sorted = zip(*sorted(zip(y_pred_probs, y_true),\
    reverse = True))
    levels = [1, 2, 5, 10, 20, 30, 50]
    for k in levels:
        evaluation_results["p_at_" + str(k)] = matric_at_k(y_true_sorted, \
        y_pred_probs_sorted, k, matric_type = "p")
        evaluation_results["r_at_" + str(k)] = matric_at_k(y_true_sorted, \
        y_pred_probs_sorted, k, matric_type = "a")

    return evaluation_results

def matric_at_k(y_true, y_pred_probs, k, matric_type):
    '''
    calculate precision/recall at different levels of population rate
    Input:
    k: integer, k percent rate treated with the intervention
    matric_type: strings, has two values of "p" and "a", which stands for
    precision and recall respectively
    Output:
    matric: float, either precision or recall at k level
    '''

    y_preds_at_k = prob_to_binary_at_k(y_pred_probs, k)
    if matric_type == "p":
        matric = precision_score(y_true, y_preds_at_k)
    elif matric_type == "a":
        matric = recall_score(y_true, y_preds_at_k)

    return matric

def prob_to_binary_at_k(y_pred_probs, k):
    '''
    To turn y_pred_probs to binary values with positive prediction as 1, and
    negative as 0
    Input:
    k: integer, the rate of population to intervene
    Output:
    a list of predictions that has binary values
    '''

    cutoff = int(len(y_pred_probs) * (k/100))
    y_pred_binary = [1 if x < cutoff else 0 for x in range(len(y_pred_probs))]
    return y_pred_binary

def plot_precision_recall(y_test, y_score, model, loop_no):
    '''
    plot the precision_recall_curve for different models
    '''
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve/{}'.format(model))
    plt.savefig('evaluation/'+ str(loop_no) + model)
    print ("plot precision_recall_curve done")
