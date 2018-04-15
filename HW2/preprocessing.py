import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing

def data_imputation(df, fill_type, vars_to_fill):
    '''
    vars_to_fill: list of strings, names of variables to imputate
    '''

    for var in vars_to_fill:
        if fill_type == "mean":
            values = df[var].mean()
        if fill_type == "zero":
            values = 0
        df[var] = df[var].fillna(value = values)

        #df.to_csv('processed_data.csv')

def clustering(df, vars):
    '''
    vars: vars usded for clustering, first being the index column, second dependent
    variable, third independent variable 1, fourth independent variable 2
    '''
    cluster = pd.concat([df[vars[0]],df[vars[1]], df[vars[2]],\
    df[vars[3]]], axis = 1)
    cluster.index = cluster[vars[0]]
    cluster.drop(cluster.columns[[0]], axis = 1, inplace = True)
    cluster_high_var1_temp = cluster[cluster[vars[2]] > df[vars[2]].mean()]
    cluster_high_var1_high_var2 = cluster_high_var1_temp[cluster_high_var1_temp[vars[3]] > df[vars[3]].mean()]
    cluster_high_var1_low_var2 = cluster_high_var1_temp[cluster_high_var1_temp[vars[3]] < df[vars[3]].mean()]
    cluster_low_var1_temp = cluster[cluster[vars[2]] < df[vars[2]].mean()]
    cluster_low_var1_high_var2 = cluster_low_var1_temp[cluster_low_var1_temp[vars[3]] > df[vars[3]].mean()]
    cluster_low_var1_low_var2 = cluster_low_var1_temp[cluster_low_var1_temp[vars[3]] < df[vars[3]].mean()]

    def get_group(x):
        if x in cluster_high_var1_high_var2.index:
            return "high_{}_high_{}".format(vars[2], vars[3])
        elif x in cluster_high_var1_low_var2.index:
            return "high_{}_low_{}".format(vars[2], vars[3])
        elif x in cluster_low_var1_high_var2.index:
            return "low_{}_high_{}".format(vars[2], vars[3])
        else:
            return "low_{}_low_{}".format(vars[2], vars[3])

    df["group"] = df[vars[0]].apply(get_group)

def gen_dummy(df, var):
    '''
    var: str, variable to be transformed into dummies
    '''
    #dummy_comluns = pd.get_dummies(df, prefix = ['dum'] * n_dummies, \
    #columns = vars)
    dummy_comluns = pd.get_dummies(df[var])
    df = pd.concat([df, dummy_comluns], axis = 1)

def discretize(df, var, bins_no, labels, method = "even_cut"):
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
