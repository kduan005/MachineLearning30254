import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy

def train_test_split(df, feature, test_size):
    '''
    feature: str, name of the dependent variable
    '''
    X = df
    Y = df[feature]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_size)

    return x_train, x_test, y_train, y_test

def build_dec_tree(x_train, y_train, x_test, y_test, depths):
    '''
    depths: a list of integer, depths used for training decision tree models
    '''
    #Lab3 for reference
    for d in depths:
        dec_tree = DecisionTreeClassifier(max_depth = d)
        dec_tree.fit(x_train, y_train)
        train_pred = dec_tree.predict(x_train)
        test_pred = dec_tree.predict(x_test)

        train_acc = accuracy(train_pred, y_train)
        test_acc = accuracy(test_pred, y_test)

        print("Depth: {} | Train acc: {:.2f} | Test acc: {:.2f}".format(d, train_acc, test_acc))

def visualize_dec_tree(df, x_train, y_train, max_depth):
    #lab3 for reference
    feature_name = y_train.columns
    class_names = pd.unique(df[feature_name])
    dec_tree = DecisionTreeClassifier(max_depth = max_depth)
    dec_tree.fit(x_train, y_train)
    viz = tree.export_graphviz(dec_tree, feature_names = x_train.columns,
                           class_names = class_names,
                           rounded = True, filled = True)
    with open("tree.dot") as f:
        dot_graph = f.read()
        graph = graphviz.Source(dot_graph)
    graph
