import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

file_path = "2020_2021_weather_combined_filled.csv"  
data = pd.read_csv(file_path, parse_dates=['time'])
start_date = "2020-01-01"
end_date = "2020-01-03"
filtered_data = data[(data['time'] >= start_date) & (data['time'] <= end_date)]
filtered_data = filtered_data.drop(columns=['Unnamed: 0'])
filtered_data.set_index('time', inplace=True)
filtered_data = filtered_data.apply(pd.to_numeric, errors='coerce')
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(
    scaler.fit_transform(filtered_data),
    columns=filtered_data.columns,
    index=filtered_data.index
)
fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
temperature_features = ['avg_module_temperature', 'avg_ambient_temperature', 'temperature_2m', 'temperature_2m_forecast']
for feature in temperature_features:
    axs[0].plot(normalized_data.index, normalized_data[feature], label=feature)
axs[0].set_title('Temperature Features')
axs[0].legend(loc='upper left')
axs[0].grid(True)
wind_features = ['avg_wind_speed', 'avg_wind_direction', 'wind_speed_10m', 'wind_speed_10m_forecast']
for feature in wind_features:
    axs[1].plot(normalized_data.index, normalized_data[feature], label=feature)
axs[1].set_title('Wind Features')
axs[1].legend(loc='upper left')
axs[1].grid(True)
precipitation_features = ['precipitation', 'precipitation_forecast', 'cloud_cover', 'cloud_cover_forecast']
for feature in precipitation_features:
    axs[2].plot(normalized_data.index, normalized_data[feature], label=feature)
axs[2].set_title('Precipitation and Cloud Cover Features')
axs[2].legend(loc='upper left')
axs[2].grid(True)

plt.tight_layout()
plt.show()
