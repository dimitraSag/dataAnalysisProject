import pandas as pd
pd.set_option('display.max_columns', None)
df = pd.read_csv('employee_data.csv', sep=';')
df2=df.copy()
print(df2.head())


print(df2)

print(df2.isnull().sum())
df2.fillna(0, inplace=True)
print(df2)

df2['HireDate'] = pd.to_datetime(df2['HireDate'])

from datetime import datetime
current_date = datetime.now() 
df2['Tenure'] = (current_date - df2['HireDate']).dt.days // 365

#df2['Tenure'] = df2['Tenure'].round(2)



df2['PerformanceLabel'] = 'Low Performer'  

quantiles = df2[df2['TotalRevenue'] > 0]['TotalRevenue'].quantile([0.5])  

df2.loc[(df2['TotalRevenue'] > quantiles[0.5]), 'PerformanceLabel'] = 'High Performer'
df2.loc[(df2['TotalRevenue'] > 0) & (df2['TotalRevenue'] <= quantiles[0.5]), 'PerformanceLabel'] = 'Average Performer'

df2.to_csv('Dataset.csv')
#values = df2['PerformanceLabel']

#series = pd.Series(values)

#count = series.value_counts()

#print(count)
