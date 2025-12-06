import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from FDNet.model import FDNet
import matplotlib.pyplot as plt
import json

DEVICE = torch.device("cpu")
BATCH_SIZE = 32
TARGET_FEATURE_INDEX = -1
HORIZONS = [1, 6, 12, 24, 48]
MODEL_SAVE_PATH = "fdnet_models/fdnet_model.pth"
NORMALIZATION_STATS_PATH = "../Data/nn_data/prepared_fdnet_data/normalization_stats.json"
DATA_DIR = "../Data/nn_data/prepared_fdnet_data"
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    X_test = np.load(os.path.join(DATA_DIR, "test_X.npy"))
    Y_test = np.load(os.path.join(DATA_DIR, "test_Y.npy"))
    return X_test, Y_test


def get_dataloader(X, Y):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def evaluate_model(model, test_loader, criterion, horizons, target_mean, target_std):
    model.eval()
    results = {h: {"loss": 0.0, "mae": 0.0, "r2": 0.0} for h in horizons}
    horizon_counts = {h: 0 for h in horizons}
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(DEVICE), batch_Y.to(DEVICE)
            output = model(batch_X)
            all_predictions.append(output.cpu().numpy())
            all_ground_truths.append(batch_Y.cpu().numpy())

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

    # Average
    for horizon in horizons:
        results[horizon]["loss"] /= horizon_counts[horizon]
        results[horizon]["mae"] /= horizon_counts[horizon]
        results[horizon]["r2"] /= horizon_counts[horizon]

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truths = np.concatenate(all_ground_truths, axis=0)
    return results, all_predictions, all_ground_truths


def main():
    X_test, Y_test = load_data()
    test_loader = get_dataloader(X_test, Y_test)
    with open(NORMALIZATION_STATS_PATH, "r") as f:
        normalization_stats = json.load(f)
    target_mean = normalization_stats["avg_dcp"]["mean"]
    target_std = normalization_stats["avg_dcp"]["std"]

    model = FDNet(enc_in=X_test.shape[-1], c_out=Y_test.shape[-1], label_len=48, pred_len=48, timebed='hour').to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    criterion = torch.nn.MSELoss()
    results, predictions, ground_truths = evaluate_model(model, test_loader, criterion, HORIZONS, target_mean, target_std)

    with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
    for horizon, metrics in results.items():
        print(f"Horizon {horizon} Hours - MSE: {metrics['loss']:.4f}, "
              f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    np.save(os.path.join(OUTPUT_DIR, "predictions.npy"), predictions)
    np.save(os.path.join(OUTPUT_DIR, "ground_truths.npy"), ground_truths)

    # Visualisation
    for h in HORIZONS:
        predictions_denorm = predictions[:, h - 1, TARGET_FEATURE_INDEX] * target_std + target_mean
        ground_truths_denorm = ground_truths[:, h - 1, TARGET_FEATURE_INDEX] * target_std + target_mean
        plt.figure(figsize=(10, 6))
        plt.plot(predictions_denorm[0:100], label="Predictions")
        plt.plot(ground_truths_denorm[0:100], label="Ground Truth")
        plt.title(f"{h}-Hour Horizon Predictions vs Ground Truth")
        plt.xlabel("Sample Index")
        plt.ylabel("Denormalized Target")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{h}_hour_predictions.png"))
        plt.close()


if __name__ == "__main__":
    main()
