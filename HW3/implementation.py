import exploration
import pandas as pd
import datetime
from datetime import datetime
import datapreprocessing
import modeling

#read in and concatenate two datasets
path_outcomes = "../outcomes/outcomes.csv"
df_outcomes = exploration.read_data(path_outcomes, index_col = "projectid")

path_projects = "../projects/projects.csv"
df_projects = exploration.read_data(path_projects, index_col = "projectid")
df_projects["date_posted"] = pd.to_datetime(df_projects["date_posted"])
date_lb = pd.Timestamp(datetime(2011, 1, 1))
date_ub = pd.Timestamp(datetime(2013, 12, 31))
date_filter = (df_projects["date_posted"] >= date_lb) & \
(df_projects["date_posted"] <= date_ub)
df_projects = df_projects[date_filter].copy()
df = df_projects.join(df_outcomes)
df = df.drop(["teacher_acctid", "schoolid", "school_ncesid", "school_latitude",\
 "school_longitude","school_city", "school_state", "school_district", \
 "school_county", "teacher_prefix"], axis = 1).copy()
#print (df.count())

#convert categorical to numerical values
categorical_vars_with_nan = ["school_metro", "primary_focus_subject", \
"primary_focus_area", "secondary_focus_subject", "secondary_focus_area", \
"resource_type", "grade_level", \
"at_least_1_teacher_referred_donor", "at_least_1_green_donation", \
"three_or_more_non_teacher_referred_donors", \
"one_non_teacher_referred_donor_giving_100_plus", \
"donation_from_thoughtful_donor"]

numerical_vars_with_nan = ["students_reached", "great_messages_proportion", \
"teacher_referred_count", "non_teacher_referred_count"]

categorical_vars = ["school_metro", "school_charter", \
"school_magnet", "school_year_round", "school_nlns", "school_kipp", \
"school_charter_ready_promise", "teacher_teach_for_america", \
"teacher_ny_teaching_fellow", "primary_focus_subject", "primary_focus_area", \
"secondary_focus_subject", "secondary_focus_area", "resource_type", \
"poverty_level", "grade_level", "eligible_double_your_impact_match", \
"eligible_almost_home_match", "is_exciting", "at_least_1_teacher_referred_donor", \
"at_least_1_green_donation", "great_chat", \
"three_or_more_non_teacher_referred_donors", \
"one_non_teacher_referred_donor_giving_100_plus", "donation_from_thoughtful_donor"]

#Split features and label
X = df.drop(["fully_funded"], axis = 1).copy()
#print ("X.columns", X.columns, X.index.values)

y = df[["fully_funded"]].copy()
datapreprocessing.encode_label(y, ["fully_funded"])
y = y.fully_funded.copy()

#Split training and testing set
splitdate = [pd.Timestamp(datetime(2012, 1, 1)), pd.Timestamp(datetime(2012, 6, 30))]
X_y_sets = modeling.split_train_test(X, y, test_size = 0.25, temporal = True,\
 temporal_var = "date_posted", split_date = splitdate)
for i, X_y_set in enumerate(X_y_sets):
    print (i)
    X_train, X_test, y_train, y_test = X_y_set

    #imputate data
    datapreprocessing.imputation(X_train, categorical_vars_with_nan, fill_type = None, value = "unknown", \
    output = False)
    datapreprocessing.imputation(X_train, numerical_vars_with_nan, fill_type = "mean", \
    value = None, output = False)
    datapreprocessing.imputation(X_test, categorical_vars_with_nan, fill_type = None, value = "unknown", \
    output = False)
    datapreprocessing.imputation(X_test, numerical_vars_with_nan, fill_type = "mean", \
    value = None, output = False)

    datapreprocessing.encode_label(X_train, categorical_vars)
    datapreprocessing.encode_label(X_test, categorical_vars)

    #Train models
    classifiers, param_grid = modeling.classifiers_parameters(grid_type = "test")
    modeling.classifier_loop(["RF", "LR", "DT", "NB", "AB", "BAG", "KNN"], \
    classifiers, param_grid, X_train, X_test, y_train, y_test, output = True, loop_no = i)
