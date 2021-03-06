import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing

def to_bools(df, vars_to_convert):
    for var in vars_to_convert:
        df[var] = df[var].astype(np.bool).copy()

def encode_label(df, vars_to_encode):
    '''
    vars_to_encode: a list of strings, variables to be transform to numerical
    labels
    '''
    le = preprocessing.LabelEncoder()
    for var in vars_to_encode:
        le.fit(df[var].unique().astype(str))
        new_column = pd.Series(le.transform(df[var].astype(str)), index = df.index.values)
        df[var] = new_column

def imputation(df, vars_to_fill, fill_type = None, value = None, \
output = False):
    '''
    fill_type: including mean or zero
    vars_to_fill: a list of strings, variables to imputate
    value_to_fill: allowing for manual input for value to be imputated. \
    fill_type and value cannot have inputs at the same time
    '''

    for var in vars_to_fill:

        if fill_type == "mean":
            value_to_fill = df[var].mean()

        elif fill_type == "zero":
            value_to_fill = 0

        else:
            value_to_fill = value
        df[var] = df[var].fillna(value_to_fill).copy()

    if output:
        df.to_csv("output/filled_missing.csv")

#clustering is not implemented in PA3
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

def discretize(df, var, bins_no = 5, size = None, bound = None, method = "even_cut"):
    '''
    var: str, the column to be discretized
    labels: a list of values assigned to different bins
    size: bin size
    bound: a list of two ints, allowing users to set upper and lower \
    bound of the evenly devided bins in middle
    method: including even cut, quartile cut, and evenly cut in the indidated \
    bound and out-of-bound given to outliers
    (https://github.com/yhat/DataGotham2013/blob/master/notebooks/7%20-%20Feature%20Engineering.ipynb)
    '''
    assert df[var].dtype != object
    discretized_var = 'discretized_' + var

    if method == "qcut":
        df[discretized_var] = pd.qcut(df[var], bins_no)
    elif method == "cut":
        if size:
            min_val = min(df[var].values)
            max_val = max(df[var].values)
            if bound:
                lowerbound, upperbound = bound
                assert lowerbound >= min_val
                assert upperbound <= max_val
                bins = [min_val] + list(range(lowerbound, upperbound, \
                size)) + [upperbound, max_val]
            else:
                bins = [min_val]
                bin_bound = min_val + size
                while bin_bound < upperbound:
                    bins.append(bin_bound)
                    bin_bound += size
            df[discretized_var] = pd.cut(df[var], bins, include_lowest = True)

        else:
            df[discretized_var] = pd.cut(df[var], bins, include_lowest = True)

def cat_to_dummy(df, vars_to_dummy, drop = True):
    '''
    generate dummy variables from categorical variables
    vars_to_dummy: a list of strings, names of the variables to convert
    '''

    for var in vars_to_dummy:
        dum_var = pd.get_dummies(df[var], var)
        df = pd.merge(df, dum_var, left_index = True, right_index = True, \
        how = "inner")

    if drop:
        df.drop(vars_to_dummy, inplace = True, axis = 1)

    return df

def normalize(df, var):
    '''
    var: str, name of the variable to be normalized
    '''
    values = df[var].values.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    var_scaled = min_max_scaler.fit_transform(values)
    df[(var + '_scaled')] = pd.DataFrame(var_scaled)
    #reference:https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
