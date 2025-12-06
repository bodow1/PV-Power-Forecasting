import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from FDNet.model import FDNet
import matplotlib.pyplot as plt

DEVICE = torch.device("cpu")
BATCH_SIZE = 32
EPOCHS = 5
INPUT_LENGTH = 48
FORECAST_LENGTH = 48 
TARGET_FEATURE_INDEX = -1
HORIZONS = [1, 6, 12, 24, 48]
DATA_DIR = "../Data/nn_data/prepared_fdnet_data"
MODEL_SAVE_DIR = "fdnet_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
NORMALIZATION_STATS_PATH = os.path.join(DATA_DIR, "normalization_stats.json")


def load_data(horizon):
    X_train = np.load(os.path.join(DATA_DIR, f"train_X.npy"))
    Y_train = np.load(os.path.join(DATA_DIR, f"train_Y.npy"))
    X_test = np.load(os.path.join(DATA_DIR, f"test_X.npy"))
    Y_test = np.load(os.path.join(DATA_DIR, f"test_Y.npy"))
    return X_train, Y_train, X_test, Y_test


def get_dataloader(X, Y):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def train_model(model, train_loader, optimizer, epochs, criterion):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(DEVICE), batch_Y.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_X)
            target_prediction = output[:, :, TARGET_FEATURE_INDEX]
            target_ground_truth = batch_Y[:, :, TARGET_FEATURE_INDEX]
            loss = criterion(target_prediction, target_ground_truth)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

def evaluate_model(model, test_loader, criterion, horizons, target_mean, target_std):
    model.eval()
    results = {h: {"loss": 0.0, "mae": 0.0, "r2": 0.0} for h in horizons}
    horizon_counts = {h: 0 for h in horizons}
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(DEVICE), batch_Y.to(DEVICE)
            output = model(batch_X)
            for horizon in horizons:
                prediction = output[:, horizon - 1, TARGET_FEATURE_INDEX]
                ground_truth = batch_Y[:, horizon - 1, TARGET_FEATURE_INDEX]
                loss = criterion(prediction, ground_truth)
                results[horizon]["loss"] += loss.item()
                prediction_denorm = prediction * target_std + target_mean
                ground_truth_denorm = ground_truth * target_std + target_mean
                mae = torch.mean(torch.abs(prediction_denorm - ground_truth_denorm))
                results[horizon]["mae"] += mae.item()
                ss_tot = torch.sum((ground_truth_denorm - torch.mean(ground_truth_denorm)) ** 2)
                ss_res = torch.sum((ground_truth_denorm - prediction_denorm) ** 2)
                r2 = 1 - ss_res / ss_tot
                results[horizon]["r2"] += r2.item()
                horizon_counts[horizon] += 1

    for horizon in horizons:
        results[horizon]["loss"] /= horizon_counts[horizon]
        results[horizon]["mae"] /= horizon_counts[horizon]
        results[horizon]["r2"] /= horizon_counts[horizon]

    return results


def plot_1h_prediction(model, test_loader, output_dir, target_mean, target_std):
    model.eval()
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(DEVICE), batch_Y.cpu()
            output = model(batch_X).cpu()
            # -1 because that is where the target feature is
            prediction = output[:, 0, -1]
            actual = batch_Y[:, 0, -1]
            prediction_denorm = prediction * target_std + target_mean
            actual_denorm = actual * target_std + target_mean

            num_hours = 24
            plt.figure(figsize=(10, 6))
            plt.plot(range(num_hours), actual_denorm[:num_hours].numpy(), label="Actual", marker="o")
            plt.plot(range(num_hours), prediction_denorm[:num_hours].numpy(), label="Prediction", marker="x")
            plt.xlabel("Hour")
            plt.ylabel("Power Output")
            plt.title("1-Hour Ahead Predictions vs Actual (24-Hour Slice)")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(output_dir, "1h_prediction_24h_slice.png"))
            plt.close()
            break  


def main():
    X_train, Y_train, X_test, Y_test = load_data(horizon=48)
    train_loader = get_dataloader(X_train, Y_train)
    test_loader = get_dataloader(X_test, Y_test)
    with open(NORMALIZATION_STATS_PATH, "r") as f:
        normalisation_stats = json.load(f)
    target_mean = normalisation_stats["avg_dcp"]["mean"]
    target_std = normalisation_stats["avg_dcp"]["std"]
    model = FDNet(enc_in=X_train.shape[-1], c_out=Y_train.shape[-1], label_len=INPUT_LENGTH, pred_len=FORECAST_LENGTH,timebed='hour').to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_model(model, train_loader, optimizer, EPOCHS, criterion)
    results = evaluate_model(model, test_loader, criterion, HORIZONS, target_mean, target_std)

    plot_1h_prediction(model=model, test_loader=test_loader, output_dir=MODEL_SAVE_DIR, target_mean=normalisation_stats["avg_dcp"]["mean"],target_std=normalisation_stats["avg_dcp"]["std"])
    for horizon, metrics in results.items():
        print(f"Horizon {horizon} Hours - MSE: {metrics['loss']:.4f}, "
            f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "fdnet_model.pth"))

if __name__ == "__main__":
    main()
