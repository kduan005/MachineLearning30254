import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from preprocessing import *

def read_data(path):
    df = pd.read_csv(path)
    return df

def plot_scatter(df, dep_var, ind_var):
    '''
    dep_var: str, name of dependent variable
    ind_var: str, name of independent variable
    '''
    plt.scatter(df[dep_var], df[ind_var])
    plt.title('{} vs {}'.format(dep_var, ind_var))
    plt.xlabel('{} coefficient'.format(ind_var)) # units unclear...
    plt.ylabel(dep_var)
    plt.show()

def plot_scatter_matrix(df, vars):
    '''
    vars: a list of variables name, the first being the outcome
    variable name, the rest being the independent variables of interest
    '''
    pd.plotting.scatter_matrix(df[vars], figsize = (12, 8))
    plt.savefig("CRRLTN_{}_OTHRS".format(vars[0]))

def corr_matrix(df, dep_var):
    '''
    dep_var: str, the name of the dependent variable
    '''
    corr_matrix = df.corr()
    print ("Correlations between {} and other variables".format(dep_var))
    print (corr_matrix[dep_var].sort_values(ascending = True))

def plot_bar(df, var, if_continuous = False, labels = None, method = None):
    '''
    var: str, the variable that to be discretized
    '''
    if if_continuous is True:
        discretize(df, var, bins_no, labels, method)
        discretized_df = df[('discretized_' + var)].value_counts()
    else:
        discretized_df = df[var].value_counts()

    discretized_df[:10].plot.bar()
    plt.show()

def plot_hist(df, var, bins_no):
    '''
    var: str, the name of the variable to plot
    '''
    plt.hist(df[var], bins = bins_no)
    plt.title('Distribution of {}'.format(var))
    plt.ylabel(var)
    plt.xlabel('Frequency')
    plt.show()
