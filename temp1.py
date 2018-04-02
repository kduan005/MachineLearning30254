import requests
from requests.auth import HTTPBasicAuth
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

#create dataframes from raw datasets
df_Alley = pd.read_csv("311_Service_Requests_-_Alley_Lights_Out.csv", error_bad_lines=False, index_col=False, dtype='unicode')
df_Graffiti = pd.read_csv("311_Service_Requests_-_Graffiti_Removal.csv", error_bad_lines=False, index_col=False, dtype='unicode')
df_Vacant = pd.read_csv("311_Service_Requests_-_Vacant_and_Abandoned_Buildings_Reported.csv", error_bad_lines=False, index_col=False, dtype='unicode')

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
plt.savefig('alley_overtime.png')

Graffiti_overtime = pd.Series(df[filter_Graffiti].groupby(pd.Grouper(key = "Creation", freq = "m")).size())
Graffiti_overtime.plot()
plt.show()
plt.savefig('graffiti_overtime.png')

Vacant_overtime = pd.Series(df[filter_Vacant].groupby(pd.Grouper(key = "Creation", freq = "m")).size())
Vacant_overtime.plot()
plt.show()
plt.savefig('vacant_overtime.png')

'''
##Statistics of requests by type and subtype
'''
all_type = df["Type"].value_counts(normalize = True)
all_type.plot.pie(autopct = '%.1f%%')
plt.show()
plt.savefig('all_type.png')

Graffiti_subtype = df[filter_Graffiti]["Subtype"].value_counts(normalize = True)
Graffiti_subtype.plot.pie(autopct = '%.1f%%')
plt.show()
plt.savefig('graffiti_subtype.png')
'''
##Statistics of requests by neighborhood
'''
all_neighborhood = df["Neighborhood"].value_counts()
all_neighborhood[:10].plot.bar()
plt.show()
plt.savefig('all_type_by_neighborhood.png')

Alley_neighborhood = df[filter_Alley]["Neighborhood"].value_counts()
Alley_neighborhood[:10].plot.bar()
plt.show()
plt.savefig('alley_by_neighborhood.png')

Graffiti_neighborhood = df[filter_Graffiti]["Neighborhood"].value_counts()
Graffiti_neighborhood[:10].plot.bar()
plt.show()
plt.savefig('graffiti_by_neighborhood.png')

Vacant_neighborhood = df[filter_Vacant]["Neighborhood"].value_counts()
Vacant_neighborhood[:10].plot.bar()
plt.show()
plt.savefig('vacant_by_neighborhood.png')
'''
##Statistics of requests by response time
'''
Alley_response = df[filter_Alley]["Response"].value_counts()
Alley_response.describe()
Alley_response[:10].plot.bar()
plt.show()
plt.savefig('alley_response.png')

Graffiti_response = df[filter_Graffiti]["Response"].value_counts()
Graffiti_response.describe()
Graffiti_response[:10].plot.bar()
plt.show()

#Vacant_response = df[filter_Vacant]["Response"].value_counts()
#Vacant_response[:10].plot.bar()
#Vacant_response.describe()
#plt.show()
'''
##Summary of Problem1

#Statistical analysis of requests over time

#From "alley_overtime.png" we can see that during 2010-2011 the request of Alley Lights Out increased sharply and dropped 
#quickly back to a reasonable level in 2013. However, a lack of data before 2010 could count for the sharp increase in 2010).
#It can be inferred from the graph that the city has launched a large scale of
#replacement after 2010, with a replacement rate of roughly 2 years since after.
#It has seen its highs and lows ever since 2010. However,
#the overall trend is in its increase.

#From "graffiti_overtime.png" we see that with the access to relevant data after 2010, the overall request of Graffiti
#Removal is at the level of 12,000 times per year. The request number fluctuates along the time, but in a steadily dcreasing
#trend.

#The graph "vacant_overtime" implies that there was a flush of vacant building reports during 2011 to 2012. Foreclosure is
#a potential explaination for the high rise of the number of vacant buildings according to 
#https://www.nytimes.com/2011/10/28/us/foreclosures-lead-to-crime-and-decay-in-abandoned-buildings.html
#It has seen a reasonable decrease after 2012, with the recovery of economy being a hypothetical correlated fact leading to
#the decline.

#Statistical analysis of types and subtypes of requests

#Of all types of requests("all_type.png), Graffiti Removal takes up 78.8%, following by Alley Light Out of 16.2% and Vacant/Abandoned Building
#of 5.0% The number is in align with the sense of existing scale of different types of issues.
#Specially, Graffiti on bricks and metal counts for roughly 60% of all the Graffiti requests (see graph "graffiti_subtype.png").

#Statistical analysis of requests by different neighborhoods

#Overall West Town is the largest source where requests come from, with South Lawndale and Logan Square in the second and
#third place respectively. Among all the major sources of reports, all the top10 neighborhoods, 8 out of 10 locate in either
#west or south side of Chicago.
#However, statistics shows that different neighborhoods have different headaches in terms of types of requests. Austin,
#Roseland, and Auburn Gresham are the major neighborhoods suffering from Alley Lights Out problems. Since Graffiti counts for
#the largest share of all requests we are looking at, it comes as no surprise that the top three cities with Graffiti issue
#are in line with the analysis when all types of requests are included. Meanwhile, Vacant/Abandoned buildings has its
#largest numbers of complaints coming from West Englewood, Englewood, Austin.

#Statistical analysis of requests by response time

#"alley_response.png" shows that more than 50% of the request could be responsed within 1 day. The longest response time 
# is 9 days. Graffiti has a similar structure of responding time with Alley Light Out, with more than 80% of the request being
# resolved within 1 day. A smaller counts of late response greater than 3 days is the major reason that counts for a 
#higher efficiency rate of the requested responed for Graffiti problems.

#Summuries of interesting findings

#Most of the requests come from West and South Chicago as expected. However, it turns out to be that people living in
#Uptown and some other middle or north Chicago neighborhood also like to complain about Graffiti.
#Graffiti and Vacant/Abandoned buildings have a similar pattern of trend in the past few years as in both see a steady
#decrease. A potential reason for that could be as the number of vacant building declines after the resilience of economy
#after 2009, as the problem of vacant buildings is tackled by the City Council, Graffiti on abandoned buildings
#has also drop to a lower level consequentially.
#However, none of the numbers of these three types of requests goes straight up or down without fluctuation. The pick of
#the request tend to appear in the middle of each year during summer time.
#All three types of requests have seen a great soar in 2011, Top results by searching "Chicago 2011" turn out to be the 
#blizzard. Futhure evidence needs to be shown to support any correlation in between.
#Only Alley Light out is having an increasing trend in terms of the numbers of requests. A possible explaination for that
#could be the increase number of street light implemented across the city.


#Problem2: Data Augmentation and APIs
filter = ((df["Creation"] > datetime.datetime(2017, 12, 31)) & (filter_Alley | \
filter_Vacant))

df_recent_month = df[filter]
df_recent_month.dropna(subset = ["ZIP"])

apikey = "5b9d4382502f66057d5e472188a11800148098ac"
request_url = "http://citysdk.commerce.gov"

df_augment = pd.DataFrame(columns = ["Type", "Creation", "Income", \
"Population Black Alone", "Education High Shool", "Employment Employed"])
'''
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
'''

#Problem3: Prediction

filter_lawndale = (df["Community"] == "29")
df_lawndale = df[filter_lawndale]
df_lawndale.groupby(pd.Grouper(key = "Type")).size()

#The data shows that for all requests coming from Lawndale, the probabilities of the three different request types are
#respectively Alley Light OUt: 27%, Graffiti Removal: 52%, Vacant/Abandoned Building: 21%. The request hence is most
#likely to be of Graffiti Removal.

filter_garfield_uptown = ((df_Graffiti["Community"] == "56")|(df_Graffiti["Community"] == "3")) 
df_Graffiti_small = df_Graffiti[filter_garfield_uptown]
df_Graffiti_small.groupby(pd.Grouper(key = "Community")).size()

#The data shows that among all 25,605 requests of Graffiti coming from either Garfield or Uptown, the requests coming from
#Garfield take up 45.4%, while the ratio of those coming from Uptown is 54.6%. Thus the call has a greater chance to come
#from Uptown than Garfield by 9.3%

#If we already know a call is of Graffiti, then in the last case the likelihood of the call coming from Garfield is
#100/(100+160)=38.5%. The likelihood of the call coming from Uptown is 160/(100+160)=61.5%. Then the call is 61.5%-38.5%=23%
#more likely to come from Uptown than Garfield.
