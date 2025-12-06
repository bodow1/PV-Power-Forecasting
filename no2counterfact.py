import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from omnixai.data.timeseries import Timeseries
from omnixai.explainers.timeseries import MACEExplainer
from FDNet.model import FDNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../Data/nn_data/prepared_fdnet_data"
MODEL_PATH = "fdnet_models/fdnet_model.pth"
NORMALIZATION_STATS_PATH = os.path.join(DATA_DIR, "normalization_stats.json")
TARGET_FEATURE = "avg_dcp"
TARGET_FEATURE_INDEX = -1
FEATURE_NAMES = [
    "avg_irradiance", "avg_module_temperature", "avg_ambient_temperature",
    "avg_wind_speed", "temperature_2m_forecast", "relative_humidity_2m_forecast",
    "precipitation_forecast", "cloud_cover_forecast", "wind_speed_10m_forecast", "avg_dcp",
]


def load_data():
    X_test = np.load(os.path.join(DATA_DIR, "test_X.npy"))
    predictions = np.load(os.path.join("evaluation_results", "predictions.npy"))
    ground_truths = np.load(os.path.join("evaluation_results", "ground_truths.npy"))
    return X_test, predictions, ground_truths


def load_model(input_dim, output_dim):
    model = FDNet(
        enc_in=input_dim, c_out=output_dim, label_len=48, pred_len=48, timebed="hour"
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def find_anomalies(predictions, ground_truths, horizon, threshold=0.1):
    deviation = np.abs(ground_truths[:, horizon - 1, TARGET_FEATURE_INDEX] - predictions[:, horizon - 1, TARGET_FEATURE_INDEX])
    anomalies = np.where(deviation > threshold)[0]
    return anomalies


def find_anomalies_raw(data, window_size=150, threshold_factor=0.1):
    rolling_mean = np.zeros_like(data)
    for i in range(len(data)):
        start_idx = max(0, i - window_size)
        end_idx = min(len(data), i + window_size + 1)
        rolling_mean[i] = np.mean(data[start_idx:end_idx])
    threshold = rolling_mean * threshold_factor
    anomalies = np.where(data < threshold)[0]
    return rolling_mean, threshold, anomalies

def is_model_correct(true_value, predicted_value, threshold=0.1):
    return (true_value - predicted_value) > threshold


def preprocess_data(X_test):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test.reshape(-1, X_test.shape[-1]))  
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    return scaler, pca, X_pca


def generate_counterfactual(model, X_test, anomaly_index, scaler, pca, horizon):
    original_data = X_test[anomaly_index, :, :]
    timestamps = pd.date_range(start="2020-01-01 00:00:00", periods=original_data.shape[0], freq="H")
    ts = Timeseries(data=pca.transform(scaler.transform(original_data.reshape(-1, X_test.shape[-1]))),timestamps=timestamps,variable_names=[f"PCA_{i}" for i in range(pca.n_components_)])
    training_ts_data = np.concatenate([X_test[i, :, :] for i in range(min(10, len(X_test)))], axis=0)
    training_ts = Timeseries(data=pca.transform(scaler.transform(training_ts_data.reshape(-1, X_test.shape[-1]))),timestamps=pd.date_range(start="2020-01-01", periods=training_ts_data.shape[0], freq="H"),variable_names=[f"PCA_{i}" for i in range(pca.n_components_)])

    def predict_function(ts_batch):
        original_features = scaler.inverse_transform(pca.inverse_transform(ts_batch.to_numpy()))
        tensor_input = torch.tensor(original_features.reshape(-1, 48, X_test.shape[-1]), dtype=torch.float32).to(DEVICE)
        outputs = model(tensor_input)
        return outputs[:, horizon - 1, TARGET_FEATURE_INDEX].detach().cpu().numpy()

    explainer = MACEExplainer(training_data=training_ts, predict_function=predict_function, mode="anomaly_detection", threshold=0.1)
    explanation = explainer.explain(ts)
    counterfactual_candidates = explainer.candidates
    return explanation, counterfactual_candidates


def plot_explanations(original_values, counterfactuals, feature_names):
    avg_counterfactual_values = []
    valid_feature_names = []
    for feature_idx, counterfactual_values in counterfactuals.items():
        if feature_idx >= len(original_values) or feature_idx >= len(feature_names):
            print(f"Skipping invalid feature index: {feature_idx}")
            continue
        avg_value = np.mean(counterfactual_values)
        avg_counterfactual_values.append(avg_value)
        valid_feature_names.append(feature_names[feature_idx])

    avg_original_values = original_values[:len(valid_feature_names)]
    x = np.arange(len(valid_feature_names))
    width = 0.35 
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, avg_original_values, width, label="Original Values", color="blue")
    plt.bar(x + width / 2, avg_counterfactual_values, width, label="Counterfactual Values", color="orange")

    plt.xticks(x, valid_feature_names, rotation=45, ha="right")
    plt.ylabel("Feature Values")
    plt.title("Feature Changes in Counterfactual Explanation")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    X_test, predictions, ground_truths = load_data()
    model = load_model(input_dim=X_test.shape[-1], output_dim=predictions.shape[-1])
    horizon = 1 # x-hour prediction
    pre_anomalies = find_anomalies_raw(predictions, ground_truths, horizon)
    anomalies = []
    horizon = 1
    for anomaly_idx in pre_anomalies:
        if is_model_correct(ground_truths[anomaly_idx, horizon - 1, TARGET_FEATURE_INDEX], predictions[anomaly_idx, horizon - 1, TARGET_FEATURE_INDEX], threshold=0.1):
            anomalies.append(anomaly_idx)
    scaler, pca, X_pca = preprocess_data(X_test)
    anomaly_index = anomalies[0]
    explanation, counterfactual_candidates = generate_counterfactual(model, X_test, anomaly_index, scaler, pca, horizon)
    original_values = X_test[anomaly_index, horizon - 1, :]
    plot_explanations(original_values, counterfactual_candidates, FEATURE_NAMES)

if __name__ == "__main__":
    main()
