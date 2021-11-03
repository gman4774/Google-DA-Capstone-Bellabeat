
# Bellabeat Case Study
### How Can a Wellness Technology Company Play It Smart?
<br>
The following is a more in depth breakdown of my case study for the second Google Data Analytics capstone project.

<br>

## 1. Data Organization
The data for this project was pulled from [here](https://www.kaggle.com/arashnic/fitbit).
<br>
This data only has 1 month of data and normally would not be enough to pull significant results from, but in this case it is all I have to work with so I continue with what I have.
<br>

I decide to do my cleaning and organizing in python, so I first import the libraries I will be using:
```python
import csv
import pandas as pd
from scipy import array, concatenate
import numpy as np


```
I decided to focus on three main files for analysis and import them:
```python
dact = pd.read_csv(r'dailyActivity_merged.csv')
sleep = pd.read_csv(r'sleepDay_merged.csv')
weight = pd.read_csv(r'weightLogInfo_merged.csv')

```
Next I organize and clean each dataset individually in case there are diffrences between them.
<br>
First I inspect the dataframe for daily activity to have a better grasp as to how it is currently organized:
```python
print(dact.head())
print(dact.columns)
print(dact.shape)

```
```
           Id ActivityDate  TotalSteps  ...  LightlyActiveMinutes  SedentaryMinutes  Calories
0  1503960366    4/12/2016       13162  ...                   328               728      1985
1  1503960366    4/13/2016       10735  ...                   217               776      1797
2  1503960366    4/14/2016       10460  ...                   181              1218      1776
3  1503960366    4/15/2016        9762  ...                   209               726      1745
4  1503960366    4/16/2016       12669  ...                   221               773      1863

[5 rows x 15 columns]
Index(['Id', 'ActivityDate', 'TotalSteps', 'TotalDistance', 'TrackerDistance',
       'LoggedActivitiesDistance', 'VeryActiveDistance',
       'ModeratelyActiveDistance', 'LightActiveDistance',
       'SedentaryActiveDistance', 'VeryActiveMinutes', 'FairlyActiveMinutes',
       'LightlyActiveMinutes', 'SedentaryMinutes', 'Calories'],
      dtype='object')
(940, 15)

```
I now have the head, column names and the shape of the data. I continue to dig a little deeper:
```python
print(dact.info())
print(dact.describe())
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 940 entries, 0 to 939
Data columns (total 15 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Id                        940 non-null    int64  
 1   ActivityDate              940 non-null    object 
 2   TotalSteps                940 non-null    int64  
 3   TotalDistance             940 non-null    float64
 4   TrackerDistance           940 non-null    float64
 5   LoggedActivitiesDistance  940 non-null    float64
 6   VeryActiveDistance        940 non-null    float64
 7   ModeratelyActiveDistance  940 non-null    float64
 8   LightActiveDistance       940 non-null    float64
 9   SedentaryActiveDistance   940 non-null    float64
 10  VeryActiveMinutes         940 non-null    int64  
 11  FairlyActiveMinutes       940 non-null    int64  
 12  LightlyActiveMinutes      940 non-null    int64  
 13  SedentaryMinutes          940 non-null    int64  
 14  Calories                  940 non-null    int64  
dtypes: float64(7), int64(7), object(1)
memory usage: 110.3+ KB
None
                 Id    TotalSteps  TotalDistance  ...  LightlyActiveMinutes  SedentaryMinutes     Calories
count  9.400000e+02    940.000000     940.000000  ...            940.000000        940.000000   940.000000
mean   4.855407e+09   7637.910638       5.489702  ...            192.812766        991.210638  2303.609574
std    2.424805e+09   5087.150742       3.924606  ...            109.174700        301.267437   718.166862
min    1.503960e+09      0.000000       0.000000  ...              0.000000          0.000000     0.000000
25%    2.320127e+09   3789.750000       2.620000  ...            127.000000        729.750000  1828.500000
50%    4.445115e+09   7405.500000       5.245000  ...            199.000000       1057.500000  2134.000000
75%    6.962181e+09  10727.000000       7.712500  ...            264.000000       1229.500000  2793.250000
max    8.877689e+09  36019.000000      28.030001  ...            518.000000       1440.000000  4900.000000

```
From this we can see the data is not ready for analysis yet, still neds to be organized a bit better. I can also see that the `ActivityDate` column is an object, not a datetime type.
<br>
I convert the `ActivityDate` column to datetime and rename column to `Date` for uniformity across all data sets.
<br>
```python
dact['Date'] = pd.to_datetime(dact['ActivityDate'])
dact['Date'] = dact['Date'].dt.strftime('%m/%d/%Y')
dact = dact.drop(columns=['ActivityDate'])
```
Next I decide to only keep the columns I'll be using for my analysis and drop the rest:
```python
dact = dact[['Id', 'Date', 'TotalSteps', 'Calories']]
```
Next I need to organize this data a bit more, so I group it by `Id` and `Date`, then get the sum of the calories and total steps for the day, for that user:
```python
dact = dact.groupby(['Id','Date'], as_index=False)['TotalSteps', 'Calories'].sum()
```
Now I print the first row to verify everything lines up the way it was intended:
```python
print(dact.iloc[0])
```
```
Id            1503960366
Date          04/12/2016
TotalSteps         13162
Calories            1985

```
Everything looks good so far, so lets move on to the next data set, `sleep` (The steps for the next two data sets are very similar to the previous one so I will only show the code, not the prints for the same segments):
```python
print(sleep.head())
print(sleep.columns)
print(sleep.shape)
print(sleep.info())
print(sleep.describe())
```
Now that I have a grasp on the sleep data structure, I begin work this data with removing columns I will not be using:
```python
sleep = sleep.drop(columns=['TotalSleepRecords', 'TotalMinutesAsleep'])
```
I verify that the columns are gone:
```python
print(sleep.columns)
print(sleep.iloc[0])
```
```
Index(['Id', 'SleepDay', 'TotalTimeInBed'], dtype='object')
Id                           1503960366
SleepDay          4/12/2016 12:00:00 AM
TotalTimeInBed                      346

```
Everything looks good so far. 
<br>
Next I trim the `Date` column since I only need the day, month and year:
```python
sleep['SleepDay'] = pd.to_datetime(sleep['SleepDay'])
sleep['Date'] = sleep['SleepDay'].dt.strftime('%m/%d/%Y')
sleep = sleep.drop(columns=['SleepDay'])
```
Then I group by `Id` and `Date`, and add up att the time in bed for that user, on that day:
```python
sleep = sleep.groupby(['Id','Date'], as_index=False)['TotalTimeInBed'].sum()
```
Now I verify everything is correct:
```python
print(sleep.iloc[0])
```
```
Id                1503960366
Date              04/12/2016
TotalTimeInBed           346

```
Great, everything looks good, time to move on to the `weight` data:
```python
print(weight.head())
print(weight.columns)
print(weight.shape)
print(weight.info())
print(weight.describe())
```
I look over the columns and determine the ones that need to be removed:
```python
weight = weight.drop(columns=['Fat', 'IsManualReport', 'LogId', 'WeightKg'])
```
Verifying that the columns are gone:
```python
print(weight.info())
print(weight.describe())
```
```
Data columns (total 4 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Id            67 non-null     int64  
 1   Date          67 non-null     object 
 2   WeightPounds  67 non-null     float64
 3   BMI           67 non-null     float64
dtypes: float64(2), int64(1), object(1)
memory usage: 2.2+ KB
None
                 Id  WeightPounds        BMI
count  6.700000e+01     67.000000  67.000000
mean   7.009282e+09    158.811801  25.185224
std    1.950322e+09     30.695415   3.066963
min    1.503960e+09    115.963147  21.450001
25%    6.962181e+09    135.363832  23.959999
50%    6.962181e+09    137.788914  24.389999
75%    8.877689e+09    187.503152  25.559999
max    8.877689e+09    294.317120  47.540001

```
Everything looks ok so far. 
<br>
I trim the `Date` column to again reduce it down to day, month and year:
```python
weight['Date'] = pd.to_datetime(weight['Date'])
weight['Date'] = weight['Date'].dt.strftime('%m/%d/%Y')
```
I organize the data by grouping it by `Id` and then `Date` and the get the mean for the remaining columns:
```python
weight = weight.groupby(['Id','Date'], as_index=False)['WeightPounds', 'BMI'].mean()
```
Now I verify everything is correct:
```python
print(weight.iloc[0])
```
```
Id              1503960366
Date            05/02/2016
WeightPounds       115.963
BMI                  22.65

```
All good, now its time for combining the data. I did an outere join for all three data sets, then sorted by `Id` then `Date`:
```python
cat = (pd.concat([dact, sleep, weight], join='outer'))
cat = cat.sort_values(['Id', 'Date'], ascending=True)
cat=  cat.reset_index(drop=True)
```
There are alot of moving parts with that step, so I look over multiple aspects of the joined data:
```python
print(cat.shape)
print(len(cat.index))
print(cat.head())
print(cat.tail())
```
```
(1417, 7)
1417
           Id        Date  TotalSteps  Calories  TotalTimeInBed  WeightPounds  BMI
0  1503960366  04/12/2016     13162.0    1985.0             NaN           NaN  NaN
1  1503960366  04/12/2016         NaN       NaN           346.0           NaN  NaN
2  1503960366  04/13/2016     10735.0    1797.0             NaN           NaN  NaN
3  1503960366  04/13/2016         NaN       NaN           407.0           NaN  NaN
4  1503960366  04/14/2016     10460.0    1776.0             NaN           NaN  NaN
              Id        Date  TotalSteps  Calories  TotalTimeInBed  WeightPounds        BMI
1412  8877689391  05/10/2016     10733.0    2832.0             NaN           NaN        NaN
1413  8877689391  05/11/2016     21420.0    3832.0             NaN           NaN        NaN
1414  8877689391  05/11/2016         NaN       NaN             NaN    188.274775  25.559999
1415  8877689391  05/12/2016      8064.0    1849.0             NaN           NaN        NaN
1416  8877689391  05/12/2016         NaN       NaN             NaN    185.188300  25.139999

```
For the moment everything went well, although still some work to do. Most of the `NaN` values cant be fixed due to the users not inputing those data points. Next we check the number of unique numbers we have in the `Id` column and then each columns type:
```python
print(cat.Id.nunique())
print(cat.dtypes)
```
```
33
Id                  int64
Date               object
TotalSteps        float64
Calories          float64
TotalTimeInBed    float64
WeightPounds      float64
BMI               float64

```
`Id` column looks good, but the `Date` column type needs to change:
```python
cat['Date'] = pd.to_datetime(cat['Date'], format='%m/%d/%Y')
```
The previous `print` from the `head()` shows us that there are multiple rows for the same `Date` and `Id`, so I merge them to individual rows and overwrite some of the `NaN` values:
```python
cat = cat.groupby(['Date', 'Id']).max()
```
Now lets verify our results:
```python
print(cat.head())
print(type(cat))
print(cat.loc['2016-04-12'])
```
```
                       TotalSteps  Calories  TotalTimeInBed  WeightPounds  BMI
Date       Id                                                                 
2016-04-12 1503960366     13162.0    1985.0           346.0           NaN  NaN
           1624580081      8163.0    1432.0             NaN           NaN  NaN
           1644430081     10694.0    3199.0             NaN           NaN  NaN
           1844505072      6697.0    2030.0             NaN           NaN  NaN
           1927972279       678.0    2220.0           775.0           NaN  NaN
<class 'pandas.core.frame.DataFrame'>
                       TotalSteps  Calories  TotalTimeInBed  WeightPounds        BMI
Date       Id                                                                       
2016-04-12 1503960366     13162.0    1985.0           346.0           NaN        NaN
           1624580081      8163.0    1432.0             NaN           NaN        NaN
           1644430081     10694.0    3199.0             NaN           NaN        NaN
           1844505072      6697.0    2030.0             NaN           NaN        NaN
           1927972279       678.0    2220.0           775.0           NaN        NaN
           2022484408     11875.0    2390.0             NaN           NaN        NaN
           2026352035      4414.0    1459.0           546.0           NaN        NaN
           2320127002     10725.0    2124.0             NaN           NaN        NaN
           2347167796     10113.0    2344.0             NaN           NaN        NaN
           2873212765      8796.0    1982.0             NaN           NaN        NaN
           3372868164      4747.0    1788.0             NaN           NaN        NaN
           3977333714      8856.0    1450.0           469.0           NaN        NaN
           4020332650      8539.0    3654.0           541.0           NaN        NaN
           4057192912      5394.0    2286.0             NaN           NaN        NaN
           4319703577      7753.0    2115.0             NaN           NaN        NaN
           4388161847     10122.0    2955.0             NaN           NaN        NaN
           4445114986      3276.0    2113.0           457.0           NaN        NaN
           4558609924      5135.0    1909.0             NaN           NaN        NaN
           4702921684      7213.0    2947.0           439.0           NaN        NaN
           5553957443     11596.0    2026.0           464.0           NaN        NaN
           5577150313      8135.0    3405.0           438.0           NaN        NaN
           6117666160         0.0    1496.0             NaN           NaN        NaN
           6290855005      4562.0    2560.0             NaN           NaN        NaN
           6775888955         0.0    1841.0             NaN           NaN        NaN
           6962181067     10199.0    1994.0           387.0    137.788914  24.389999
           7007744171     14172.0    2937.0             NaN           NaN        NaN
           7086361926     11317.0    2772.0           525.0           NaN        NaN
           8053475328     18060.0    3186.0             NaN           NaN        NaN
           8253242879      9033.0    2044.0             NaN           NaN        NaN
           8378563200      7626.0    3635.0           356.0           NaN        NaN
           8583815059      5014.0    2650.0             NaN           NaN        NaN
           8792009665      2564.0    2044.0           493.0           NaN        NaN
           8877689391     23186.0    3921.0             NaN    189.156628  25.680000

```
The grouping worked well, so lets move on to exporting for analysis in Tableau:
```python
cat.to_csv("cat.csv", index=True, encoding='utf-8-sig')
```
Now we are all set to move on to Tableau.
## 2. Analysis / Tableau
I attempt to compare the dates against the calories burned, and then dates against the steps taken, and see the data ins't showing me too much at first. I was able to determine there was a corelaton between calories burned and total steps taken.
<br>
I decide to consolidate the data to day of the week vs calories and again day of week vs steps. NI was able to produce the following two tree maps to better display visually what the data was telling me:
<br>
![Average number of steps taken per week day](https://raw.githubusercontent.com/gman4774/Google-DA-Capstone-Bellabeat/main/steps.png)
<br>
![Average number of calories users burned per week day](https://raw.githubusercontent.com/gman4774/Google-DA-Capstone-Bellabeat/main/calories.png)
<br>
![Both combined](https://raw.githubusercontent.com/gman4774/Google-DA-Capstone-Bellabeat/main/Dashboard%201%20(1).png)
<br>
The data shows me that the most active days are Tuesday and Saturday and the least active days are Sunday and Thursday.
I export the images for the presentation in google sheets.


## 3. Presentation
Everything was compiled into a short presentation [here](https://github.com/gman4774/Google-DA-Capstone-Bellabeat/blob/main/presentation_2.pptx).

