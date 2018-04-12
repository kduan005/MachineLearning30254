import pandas as pd
import numpy as np
import csv

def data_imputation(df, fill_type, vars_to_fill, output = True):
    values = {}

    for var in vars_to_fill:
        if fill_type = "mean":
            values[var] = df[var].mean()
        if fill_type = "zero":
            values[var] = 0

    df.fillna(value = values)

    if not output:
        df.to_csv('output/processed_data.csv')

def gen_dummy(df, vars):
    '''
    vars: a list of vars to generate dummies
    '''
    dummy_comluns = pd.get_dummies(df, prefix = ['dum'], columns = vars)
    df = pd.concat([df, dummy_comluns], axis = 1)

def discretize(df, var, bins_no, labels, method = None):
    '''
    var: str, the column to be discretized
    labels: a list of values assigned to different bins
    '''
    if method == 'even_cut':
        df[('discretized_' + var)] = pd.cut(df[var], bins_no, labels = labels, right = True,\
        include_lowest = True)
