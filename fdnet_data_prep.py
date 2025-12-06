import os
import numpy as np
import pandas as pd

BACKCAST_LENGTH = 48
FORECAST_LENGTH = 48
FEATURES = [
    'avg_irradiance', 'avg_module_temperature', 'avg_ambient_temperature',
    'avg_wind_speed', 'avg_wind_direction', 'temperature_2m_forecast',
    'relative_humidity_2m_forecast', 'precipitation_forecast',
    'cloud_cover_forecast', 'wind_speed_10m_forecast',
    'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'avg_dcp'
]
TARGET_COL = 'avg_dcp'


def add_time_features(df):
    df['hour_of_day'] = pd.to_datetime(df['time']).dt.hour
    df['day_of_year'] = pd.to_datetime(df['time']).dt.dayofyear
    df['sin_hour'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    return df.drop(columns=['hour_of_day', 'day_of_year', "Unnamed: 0"])


# Do this here (don't forget to denormalise later)
def normalize_features(df, train_stats=None):
    if train_stats is None:
        stats = {'mean': df.mean(), 'std': df.std()}
    else:
        stats = train_stats
    normalized_df = (df - stats['mean']) / stats['std']
    return normalized_df, stats


def prepare_sliding_window(data, backcast_length, forecast_length):
    X, Y = [], []
    for i in range(len(data) - backcast_length - forecast_length + 1):
        X.append(data.iloc[i:i + backcast_length].values)
        Y.append(data.iloc[i + backcast_length:i + backcast_length + forecast_length].values)
    return np.array(X), np.array(Y)


def main(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)
    df = add_time_features(df)
    df = df[FEATURES]
    n = len(df)
    train_end = int(0.8 * n)
    # Not using validation actually so ignore all val things :)
    val_end = int(0.8 * n)
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    train_normalized, train_stats = normalize_features(train_df)
    val_normalized, _ = normalize_features(val_df, train_stats)
    test_normalized, _ = normalize_features(test_df, train_stats)
    np.save(os.path.join(output_dir, 'normalization_stats.npy'), train_stats)

    train_X, train_Y = prepare_sliding_window(train_normalized, BACKCAST_LENGTH, FORECAST_LENGTH)
    val_X, val_Y = prepare_sliding_window(val_normalized, BACKCAST_LENGTH, FORECAST_LENGTH)
    test_X, test_Y = prepare_sliding_window(test_normalized, BACKCAST_LENGTH, FORECAST_LENGTH)

    np.save(os.path.join(output_dir, 'train_X.npy'), train_X)
    np.save(os.path.join(output_dir, 'train_Y.npy'), train_Y)
    np.save(os.path.join(output_dir, 'val_X.npy'), val_X)
    np.save(os.path.join(output_dir, 'val_Y.npy'), val_Y)
    np.save(os.path.join(output_dir, 'test_X.npy'), test_X)
    np.save(os.path.join(output_dir, 'test_Y.npy'), test_Y)

main("../2020_2021_weather_combined_filled.csv", "prepared_fdnet_data")
