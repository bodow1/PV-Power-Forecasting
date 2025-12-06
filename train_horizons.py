# Didn't have space to include the results from this model in the project paper, but it was a complex 
# yet interpretable linear model to do the same thing.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.DLinear import Model


class Configs:
    def __init__(self, seq_len, pred_len, enc_in, individual):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual


# Training function
def train_model(train_X, train_y, val_X, val_y, config, save_path, device):
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_X, dtype=torch.float32), torch.tensor(val_y, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = Model(config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    for epoch in range(50):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            output = output[:, :, -1]
            y_batch = y_batch.unsqueeze(-1)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                output = output[:, :, -1]
                y_batch = y_batch.unsqueeze(-1)
                val_loss += criterion(output, y_batch).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 48  
    enc_in = 15  
    individual = False  
    horizons = [1, 6, 12, 24, 48] 
    data_dir = "../Data/nn_data/prepared_nn_data/" 
    model_dir = "models/solar_data/" 
    os.makedirs(model_dir, exist_ok=True)
    
    for horizon in horizons:
        train_X = np.load(os.path.join(data_dir, f'train_X_h{horizon}.npy'))
        train_y = np.load(os.path.join(data_dir, f'train_y_h{horizon}.npy'))
        val_X = np.load(os.path.join(data_dir, f'val_X_h{horizon}.npy'))
        val_y = np.load(os.path.join(data_dir, f'val_y_h{horizon}.npy'))
        pred_len = 1
        config = Configs(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, individual=individual)
        save_path = os.path.join(model_dir, f'model_h{horizon}.pt')
        train_model(train_X, train_y, val_X, val_y, config, save_path, device)

if __name__ == "__main__":
    main()
