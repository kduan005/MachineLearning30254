import pandas as pd
import numpy as np
import exploration
import preprocessing
import modeling

#reading data
path = "credit-data.csv"
df = exploration.read_data(path)

#imputating holes
preprocessing.data_imputation(df, "mean", ["MonthlyIncome"])

#discretizing variables of interest
preprocessing.discretize(df, "RevolvingUtilizationOfUnsecuredLines", 10, \
list(range(1,11)))
preprocessing.discretize(df, "NumberOfOpenCreditLinesAndLoans", 5, \
list(range(1,6)))

#normalizing variables
preprocessing.normalize(df, "MonthlyIncome")
#clustering
preprocessing.clustering(df, ["PersonID", "SeriousDlqin2yrs", "RevolvingUtilizationOfUnsecuredLines", "DebtRatio"])
#generating dummny variables
preprocessing.gen_dummy(df, "group")
