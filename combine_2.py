import csv
import pandas as pd
from scipy import array, concatenate
import numpy as np


dact = pd.read_csv(r'dailyActivity_merged.csv')
sleep = pd.read_csv(r'sleepDay_merged.csv')
weight = pd.read_csv(r'weightLogInfo_merged.csv')


print('DAILY ACTIVITY DATA')
print(dact.head())
print(dact.columns)
print(dact.shape)
print(dact.info())
print(dact.describe())

#I wanted to break up the describe function to each individual column. I iterated over the columns and called describe() for each 
for x in dact.columns:
	print(dact[x].describe())
#I convert the date column from object to datetime and rename column to Date for easier data manipulation
dact['Date'] = pd.to_datetime(dact['ActivityDate'])
dact['Date'] = dact['Date'].dt.strftime('%m/%d/%Y')
dact = dact.drop(columns=['ActivityDate'])
#next I keep only the columns I'll need
dact = dact[['Id', 'Date', 'TotalSteps', 'Calories']]
#I group by Id and Date, then get the sum of the calories and total steps for that day, for that user
dact = dact.groupby(['Id','Date'], as_index=False)['TotalSteps', 'Calories'].sum()
#verifying everything is tidy
print('tidy 0')
print(dact.iloc[0])

#now i'll inspect the other two dataframes i'll be using
#sleep dataframe
print('SLEEP DATA')
print(sleep.head())
print(sleep.columns)
print(sleep.shape)
print(sleep.info())
print(sleep.describe())

#I iterated over the columns and called describe() for each
for x in sleep.columns:
	print(sleep[x].describe())
#I decide to keep Id, SleepDay and TotalTimeInBed
sleep = sleep.drop(columns=['TotalSleepRecords', 'TotalMinutesAsleep'])

print(sleep.columns)
print(sleep.iloc[0])
#I trim the date since we dont need the hours, min and sec
sleep['SleepDay'] = pd.to_datetime(sleep['SleepDay'])
sleep['Date'] = sleep['SleepDay'].dt.strftime('%m/%d/%Y')
sleep = sleep.drop(columns=['SleepDay'])

#I decide to group by Id and Date, and add up all the time in bed for that user, on that day
sleep = sleep.groupby(['Id','Date'], as_index=False)['TotalTimeInBed'].sum()
#verifying everything is tidy
print('tidy 1')
print(type(sleep))
print(sleep)
print(sleep.iloc[0])


#weight dataframe
print('WEIGHT DATA')
print(weight.head())
print(weight.columns)
print(weight.shape)
print(weight.info())
print(weight.describe())
#info shows each column has 67 non-nulls except for Fat column, upon closer inspection there are only two entry in the column so I'll remove it
#IsManualReported is only letting us know if the entry was manually put in, or automatically through the devices. This will also be removed.
#LogId does not help with our analysis, will also remove this column
#There are two columns for weight, I will be using the WeightPounds so the WeightKg will be removed
weight = weight.drop(columns=['Fat', 'IsManualReport', 'LogId', 'WeightKg'])
#verifying that the columns were removed
print(weight.info())
print(weight.describe())
#I iterated over the columns and called describe() for each
for x in weight.columns:
	print(weight[x].describe())
#I trim the date since we dont need the hours, min and sec
weight['Date'] = pd.to_datetime(weight['Date'])
weight['Date'] = weight['Date'].dt.strftime('%m/%d/%Y')
#this dataframe will be grouped same as the sleep dataframe, by Id and Date. I create the mean for the day if theres multiple inputs for that day
weight = weight.groupby(['Id','Date'], as_index=False)['WeightPounds', 'BMI'].mean()
#verifying everything is tidy
print('tidy 2')
print(weight.iloc[0])

cat = (pd.concat([dact, sleep, weight], join='outer'))
cat = cat.sort_values(['Id', 'Date'], ascending=True)
cat=  cat.reset_index(drop=True)
print(cat.shape)
print(len(cat.index))
print(cat.head())
print(cat.tail())

print(cat.Id.nunique())
print(cat.dtypes)

cat['Date'] = pd.to_datetime(cat['Date'], format='%m/%d/%Y')
#i noticed at this point I had multiple rows for the same date and same users, I merged them and had the data over NaN values
cat = cat.groupby(['Date', 'Id']).max()

print(cat.head())
print(type(cat))
print(cat.loc['2016-04-12'])

#exporting to csv for further analysis
cat.to_csv("cat.csv", index=True, encoding='utf-8-sig')



















