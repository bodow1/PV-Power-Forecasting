import pandas as pd

# Convert from minutes to hourly
df = pd.read_csv('jan_sep_2022_1minute.csv', header=0)
df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0']) 
df.set_index('Unnamed: 0', inplace=True)
df_hourly = df.resample('H').first()
df_hourly.to_csv('jan_sep_2022_hourly.csv')

