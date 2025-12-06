import pandas as pd

def process_data(file_name, output_name):
    df_hourly = pd.read_csv(file_name + '.csv', index_col=0)

    # New guys
    df_hourly['avg_dcp'] = df_hourly[['row2dcp']]
    df_hourly['avg_kWh'] = df_hourly[['row2kWh']]
    df_hourly['avg_irradiance'] = df_hourly[['row2Gfront', 'row3Gfront', 'row7Gfront']].mean(axis=1)
    df_hourly['avg_module_temperature'] = df_hourly[['row2tmod_1', 'row4tmod_1', 'row9tmod_1']].mean(axis=1)
    df_hourly['avg_ambient_temperature'] = df_hourly[['row2temperature_ambient', 'row5temperature_ambient', 'row8temperature_ambient']].mean(axis=1)
    df_hourly['avg_wind_speed'] = df_hourly[['row2wind_speed', 'row7wind_speed']].mean(axis=1)
    df_hourly['avg_wind_direction'] = df_hourly[['row2wind_direction', 'row7wind_direction']].mean(axis=1)
    df_hourly = df_hourly[['avg_dcp', 'avg_kWh', 'avg_irradiance', 'avg_module_temperature', 'avg_ambient_temperature', 'avg_wind_speed', 'avg_wind_direction']]

    df_hourly.to_csv(output_name + '.csv')

process_data('year_2020_hourly', '2020_merged')
process_data('year_2021_hourly', '2021_merged')
df_2020 = pd.read_csv('2020_weather_merged.csv')
df_2021 = pd.read_csv('2021_weather_merged.csv')
df_combined = pd.concat([df_2020, df_2021], axis=0, ignore_index=True)
df_combined.to_csv('2020_2021_weather_combined.csv', index=False)

