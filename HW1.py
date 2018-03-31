import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

#files = glob.glob('/Users/duanke/Desktop/Machine Learning/HW/HW1/*.csv')

#def get_merged(files, **kwargs):
#    df = pd.read_csv(files[0], **kwargs)
#    for f in files[1:]:
#        df = df.merge(pd.read_csv(f, **kwargs), how ='outer')
#    return df

#df = get_merged(files)

current_path = os.getcwd()
df1 = pd.read_csv(current_path + "\\311_Service_Requests_-_Alley_Lights_Out_Sample.csv")
#df2 = pd.read_csv(current_path + "\\311_Service_Requests_-_Graffiti_Removal_Sample.csv")
df3 = pd.read_csv(current_path + "\\311_Service_Requests_-_Vacant_and_Abandoned_Buildings_Reported_Sample.csv")

df3 = df3.rename(columns = {"SERVICE REQUEST TYPE": "Type of Service Request", "DATE SERVICE REQUEST WAS RECEIVED": "Creation Date"})
df = pd.concat([df1,df2, df3], axis=0, ignore_index=True)

#Problem1:
df.groupby("Type of Service Request").size()
df["Creation Date"] = pd.to_datetime(df["Creation Date"])
df["Creation Year"] = df["Creation Date"].year
df["Creation Month"] = df["Creation Date"].month
df["Creation MonthYear"] = df["Month"].astype(str) + df["Year"].astype(str)
request_type_over_time = df.groupby(("Type of Service Request", "Creation MonthYear")).size()
request_type_neignborhood = df.groupby(("Type of Service Request", "Community Area")).size()
df["Reponse Time"]= df["Completion Date"] - df["Creation Date"]
request_type_response_time =df.groupby(("Type of Service Request", "Reponse Time"))
