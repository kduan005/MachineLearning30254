import requests
from requests.auth import HTTPBasicAuth
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

#create dataframes from raw datasets
df_Alley = pd.read_csv("311_Service_Requests_-_Alley_Lights_Out_Sample.csv", error_bad_lines=False, index_col=False, dtype='unicode')
df_Graffiti = pd.read_csv("311_Service_Requests_-_Graffiti_Removal_Sample.csv", error_bad_lines=False, index_col=False, dtype='unicode')
df_Vacant = pd.read_csv("311_Service_Requests_-_Vacant_and_Abandoned_Buildings_Reported_Sample.csv", error_bad_lines=False, index_col=False, dtype='unicode')

#extract attributes in interest
df_Alley = df_Alley[["Creation Date", "Completion Date", "Type of Service Request", "Community Area", "ZIP Code"]]
df_Alley.columns = ["Creation", "Completion", "Type", "Community", "ZIP"]

df_Graffiti = df_Graffiti[["Creation Date", "Completion Date", "Type of Service Request", "What Type of Surface is the Graffiti on?", "Community Area", "ZIP Code"]]
df_Graffiti.columns = ["Creation", "Completion", "Type", "Subtype", "Community", "ZIP"]

df_Vacant = df_Vacant[["DATE SERVICE REQUEST WAS RECEIVED", "SERVICE REQUEST TYPE", "Community Area", "ZIP CODE"]]
df_Vacant.columns = ["Creation", "Type", "Community", "ZIP"]

#concatenate dataframes into a single dataframe
df = pd.concat([df_Alley, df_Graffiti, df_Vacant])
df["Community"] = df["Community"].astype(str)

#calculate response time
df["Creation"] = pd.to_datetime(df["Creation"])
df["Completion"] = pd.to_datetime(df["Completion"])
df["Response"] = df["Completion"] - df["Creation"]

#mapping comminity area to neighborhood
neighborhood = pd.read_csv("CommAreas.csv")
neighborhood = neighborhood[["AREA_NUM_1", "COMMUNITY"]]
neighborhood.columns = ["Community", "Neighborhood"]
neighborhood["Community"] = neighborhood["Community"].astype(str)
df = pd.DataFrame.merge(df, neighborhood, on = "Community")
df["Year"] = df["Creation"].dt.year

#Problem1: Data Acquisition and Analysis
##Statistics of different types of requests over time
filter_Alley = df["Type"] == "Alley Light Out"
filter_Graffiti = df["Type"] == "Graffiti Removal"
filter_Vacant = df["Type"] == "Vacant/Abandoned Building"

#all_type = df.groupby(pd.Grouper(key = "Creation", freq = "m")).size()
#plt.plot(all_type, label = "All Request Type")
#plt.savefig("all_type.png")
'''
Alley_overtime = pd.Series(df[filter_Alley].groupby(pd.Grouper(key = "Creation", freq = "m")).size())
Alley_overtime.plot()
plt.show()

Graffiti_overtime = pd.Series(df[filter_Graffiti].groupby(pd.Grouper(key = "Creation", freq = "m")).size())
Graffiti_overtime.plot()
plt.show()

Vacant_overtime = pd.Series(df[filter_Vacant].groupby(pd.Grouper(key = "Creation", freq = "m")).size())
Vacant_overtime.plot()
plt.show()
'''
##Statistics of requests by type and subtype
'''
all_type = df["Type"].value_counts(normalize = True)
all_type.plot.pie(autopct = '%.1f%%')
plt.show()

Graffiti_subtype = df[filter_Graffiti]["Subtype"].value_counts(normalize = True)
Graffiti_subtype.plot.pie(autopct = '%.1f%%')
plt.show()
'''
##Statistics of requests by neighborhood
'''
all_neighborhood = df["Neighborhood"].value_counts()
all_neighborhood[:10].plot.bar()
plt.show()

Alley_neighborhood = df[filter_Alley]["Neighborhood"].value_counts()
Alley_neighborhood[:10].plot.bar()
plt.show()

Graffiti_neighborhood = df[filter_Graffiti]["Neighborhood"].value_counts()
Graffiti_neighborhood[:10].plot.bar()
plt.show()

Vacant_neighborhood = df[filter_Vacant]["Neighborhood"].value_counts()
Vacant_neighborhood[:10].plot.bar()
plt.show()
'''
##Statistics of requests by response time
'''
#Alley_response = df[filter_Alley]["Response"].value_counts()
#Alley_response[:10].plot.bar()
#plt.show()

Graffiti_response = df[filter_Graffiti]["Response"].value_counts()
Graffiti_response[:10].plot.bar()
plt.show()

#Vacant_response = df[filter_Vacant]["Response"].value_counts()
#Vacant_response[:10].plot.bar()
#plt.show()
'''
#Summary of Problem1

#Problem2: Data Augmentation and APIs
filter = ((df["Creation"] > datetime.datetime(2017, 12, 31)) & (filter_Alley | \
filter_Vacant))

df_recent_month = df[filter]
df_recent_month.dropna(subset = ["ZIP"])

apikey = "5b9d4382502f66057d5e472188a11800148098ac"
request_url = "http://citysdk.commerce.gov"

df_augment = pd.DataFrame(columns = ["Type", "Creation", "Income", \
"Population Black Alone", "Education High Shool", "Employment Employed"])

for row in df_recent_month["ZIP", "Year", "Type"].itertuples():
    request_obj = {
      'zip': row[1],
      'state': 'IL',
      'level': 'blockGroup',
      'sublevel': False,
      'api': 'acs5',
      'year': row[2],
      'variables': ['income', 'population_black_alone', 'education_high_school',\
       'employment_employed']
    }
    response = requests.post(request_url, auth = HTTPBasicAuth(apikey, None), json = request_obj)
    if response.json()["data"]["geographyValidForAPI"] is True:
        data = response.json()["data"][0]
        income = data["B19013_001E"]
        population_black_alone = data["B02001_003E"]
        education_high_school = data["B15003_017E"]
        employment_employed = data["B23025_004E"]
        df_augment = df_new_var.append({"Type of Service Request": row[3],\
                                        "YearMonth": row[4],\
                                        "Income": income,\
                                        "Population Black Alone": population_black_alone,\
                                        "Education High Shool": education_high_school,\
                                        "Employment Employed": employment_employed},
                                        ignore_index = True)
