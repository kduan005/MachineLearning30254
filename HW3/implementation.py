import exploration
import pandas as pd
import datetime
from datetime import datetime

#read in and concatenate two datasets
path_outcomes = "../outcomes/outcomes.csv"
df_outcomes = exploration.read_data(path_outcomes, index_col = "projectid")
path_projects = "../projects/projects.csv"
df_projects = exploration.read_data(path_projects, index_col = "projectid")
df_projects["date_posted"] = pd.to_datetime(df_projects["date_posted"])
date_lb = pd.Timestamp(datetime(2012, 1, 1))
date_ub = pd.Timestamp(datetime(2013, 12, 31))
date_filter = (df_projects["date_posted"] >= date_lb) & \
(df_projects["date_posted"] <= date_ub)
df_projects = df_projects[date_filter].copy()
df = df_projects.join(df_outcomes)
df = df.drop(["teacher_acctid", "schoolid", "school_ncesid", "school_latitude",\
 "school_longitude","school_city", "school_state", "school_district", \
 "school_county", "teacher_prefix", "date_posted"], axis = 1).copy()

 
