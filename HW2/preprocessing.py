import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing

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
    if (not method) or (method == 'even_cut'):
        df[('discretized_' + var)] = pd.cut(df[var], bins_no, labels = labels, right = True,\
        include_lowest = True)

def normalize(df, var):
    '''
    var: str, name of the variable to be normalized
    '''
    values = df[var].values.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    var_scaled = min_max_scaler.fit_transform(values)
    df[(var + '_scaled')] = pd.DataFrame(var_scaled)
    #reference:https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
