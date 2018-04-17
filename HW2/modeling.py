import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy

def split(df, dep_var, drop_ind_vars, test_size):
    '''
    drop_ind_vars: vars that need to drop
    dep_var: str, name of the dependent variable
    '''
    X = df.drop(drop_ind_vars, axis = 1)

    Y = df[dep_var].astype(bool)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_size)

    return x_train, x_test, y_train, y_test

def build_dec_tree(x_train,  x_test, y_train, y_test):
    #Lab3 for reference
    for d in [1, 3, 5, 7]:
        dec_tree = DecisionTreeClassifier(max_depth = d)
        dec_tree.fit(x_train, y_train)
        train_pred = dec_tree.predict(x_train)
        test_pred = dec_tree.predict(x_test)

        train_acc = accuracy(train_pred, y_train)
        test_acc = accuracy(test_pred, y_test)

        print("Depth: {} | Train acc: {:.2f} | Test acc: {:.2f}".format(d, train_acc, test_acc))
