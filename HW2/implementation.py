import pandas as pd
import numpy as np
import exploration
import preprocessing
import modeling


#reading data
path = "credit-data.csv"
df = exploration.read_data(path, index_col = "PersonID")

#imputating holes
preprocessing.data_imputation(df, "mean", ["MonthlyIncome"])
preprocessing.data_imputation(df, "mean", ["NumberOfDependents"])

#discretizing variables of interest
preprocessing.discretize(df, "age", size = 5, bound = (22, 80), method = "cut")
preprocessing.discretize(df, "DebtRatio", bins_no = 4, method = "qcut")

'''
#clustering
preprocessing.clustering(df, ["PersonID", "SeriousDlqin2yrs", \
"RevolvingUtilizationOfUnsecuredLines", "DebtRatio"])
'''

#generate dummy
dummy_comluns1 = pd.get_dummies(df["discretized_DebtRatio"], prefix = "discretized_DebtRatio")
dummy_comluns2 = pd.get_dummies(df["discretized_age"], prefix = "discretized_age")
df = pd.concat([df, dummy_comluns1, dummy_comluns2], axis = 1)

#exploration
vars = ["age", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate", \
"NumberRealEstateLoansOrLines"]
for var in vars:
    exploration.plot_hist(df, var, 10)

exploration.plot_scatter_matrix(df, ["SeriousDlqin2yrs", \
"RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio", "MonthlyIncome"])
exploration.corr_matrix(df, "SeriousDlqin2yrs")

#modeling
x_train, x_test, y_train, y_test = modeling.split(df, \
"SeriousDlqin2yrs", ["age", "DebtRatio", "SeriousDlqin2yrs", "discretized_age", "discretized_DebtRatio"], test_size = 0.3)

modeling.build_dec_tree(x_train, x_test, y_train, y_test)
