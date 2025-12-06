import pandas as pd
import numpy as np

def create_dataset_with_interval(input_file, output_file, interval_hours):
    df = pd.read_csv(input_file + '.csv')
    df['time'] = pd.to_datetime(df['Unnamed: 0'])
    df['hour_of_day'] = df['time'].dt.hour
    df['day_of_year'] = df['time'].dt.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['target_dcp'] = df['avg_dcp'].shift(-interval_hours)
    df['temperature_2m_forecast'] = df['temperature_2m_forecast'].shift(-interval_hours)
    df['relative_humidity_2m_forecast'] = df['relative_humidity_2m_forecast'].shift(-interval_hours)
    df['precipitation_forecast'] = df['precipitation_forecast'].shift(-interval_hours)
    df['cloud_cover_forecast'] = df['cloud_cover_forecast'].shift(-interval_hours)
    df['wind_speed_10m_forecast'] = df['wind_speed_10m_forecast'].shift(-interval_hours)
    df['lag_1_dcp'] = df['avg_dcp'].shift(interval_hours)
    df['lag_2_dcp'] = df['avg_dcp'].shift(2 * interval_hours)
    df['lag_3_dcp'] = df['avg_dcp'].shift(3 * interval_hours)
    df['lag_4_dcp'] = df['avg_dcp'].shift(4 * interval_hours)
    # Not included guys
    df = df.drop(columns=['time', 'Unnamed: 0', 'temperature_2m', 'relative_humidity_2m', 'precipitation',
                          'cloud_cover', 'wind_speed_10m', 'day_of_year', 'hour_of_day'])
    # To clean up a bit
    df = df.dropna()

    df.to_csv(output_file + '.csv', index=False)

def split_dataset(input_file, output_file_train, output_file_test):
    df = pd.read_csv(input_file + '.csv')
    n = len(df)
    train_size = int(0.8 * n)
    train_data = df[:train_size]
    test_data = df[train_size:]
    train_data.to_csv(output_file_train + '.csv', index=False)
    test_data.to_csv(output_file_test + '.csv', index=False)


create_dataset_with_interval('../2020_2021_weather_combined_filled', '48_hour_lr_data', interval_hours=48)
create_dataset_with_interval('../2020_2021_weather_combined_filled', '24_hour_lr_data', interval_hours=24)
create_dataset_with_interval('../2020_2021_weather_combined_filled', '12_hour_lr_data', interval_hours=12)
create_dataset_with_interval('../2020_2021_weather_combined_filled', '6_hour_lr_data', interval_hours=6)
create_dataset_with_interval('../2020_2021_weather_combined_filled', '1_hour_lr_data', interval_hours=1)   

split_dataset('48_hour_lr_data', 'train_48_hour_lr_data', 'test_48_hour_lr_data')
split_dataset('24_hour_lr_data', 'train_24_hour_lr_data', 'test_24_hour_lr_data')
split_dataset('12_hour_lr_data', 'train_12_hour_lr_data', 'test_12_hour_lr_data')
split_dataset('6_hour_lr_data', 'train_6_hour_lr_data', 'test_6_hour_lr_data')
split_dataset('1_hour_lr_data', 'train_1_hour_lr_data', 'test_1_hour_lr_data')

