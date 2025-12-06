import pandas as pd
import numpy as np


def add_noise(df, feature_name, window):
    rolling_std = df[feature_name].rolling(window=window, min_periods=1, center=True).std()
    noise = np.random.normal(0, rolling_std)
    return df[feature_name] + noise

def add_weather_data(df_file, weather_file, output_file, window=10):
    df = pd.read_csv(df_file + '.csv')
    weather = pd.read_csv(weather_file + '.csv', parse_dates=['time'])
    df['time'] = pd.to_datetime(df['Unnamed: 0'])
    df['time'] = df['time'].dt.tz_localize(None)
    weather['time'] = weather['time'].dt.tz_localize(None)
    df_merged = pd.merge(df, weather, on='time', how='left')

    df_merged['temperature_2m_forecast'] = add_noise(df_merged, 'temperature_2m', window)
    df_merged['relative_humidity_2m_forecast'] = add_noise(df_merged, 'relative_humidity_2m', window)
    df_merged['precipitation_forecast'] = add_noise(df_merged, 'precipitation', window)
    df_merged['cloud_cover_forecast'] = add_noise(df_merged, 'cloud_cover', window)
    df_merged['wind_speed_10m_forecast'] = add_noise(df_merged, 'wind_speed_10m', window)


    df_merged = df_merged.drop(columns=['time'])
    df_merged.to_csv(output_file + '.csv', index=False)


add_weather_data('2020_merged', '2020_weather', '2020_weather_merged_forecast')
add_weather_data('2021_merged', '2021_weather', '2021_weather_merged_forecast')