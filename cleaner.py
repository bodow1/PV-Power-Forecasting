import pandas as pd

def fill_with_previous_day(df):
    df_shifted = df.shift(periods=1440)
    df_filled = df.fillna(df_shifted)
    df_filled = df_filled.fillna(method='ffill')
    return df_filled

df_combined = pd.read_csv('2020_2021_weather_combined.csv')
df_combined['time'] = pd.to_datetime(df_combined['Unnamed: 0'])
df_combined_filled = fill_with_previous_day(df_combined)
df_combined_filled.to_csv('2020_2021_weather_combined_filled.csv', index=False)
