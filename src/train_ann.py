import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from utils import DATA_DIR, MODELS_DIR, IMAGES_DIR, log_info, log_success, set_seed

# Define PyTorch Dataset
class DiabetesDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = torch.tensor(df.drop(columns=["Outcome"]).values, dtype=torch.float32)
        self.y = torch.tensor(df["Outcome"].values, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define Custom ANN Architecture
class DiabetesANN(nn.Module):
    def __init__(self, input_dim):
        super(DiabetesANN, self).__init__()
        
        self.network = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Hidden Layer 2
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden Layer 3
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output Layer (No sigmoid since we use BCEWithLogitsLoss)
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def train_ann(seed=42, epochs=300, batch_size=32, lr=0.005, patience=25):
    set_seed(seed)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"Using device: {device}")
    
    # Load dataset splits
    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "val.csv")
    
    train_dataset = DiabetesDataset(train_path)
    val_dataset = DiabetesDataset(val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = train_dataset.X.shape[1]
    
    # Instantiate model, loss, and optimizer
    model = DiabetesANN(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine annealing scheduler for smooth learning rate decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    log_info("Starting Neural Network training...")
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * X_batch.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item() * X_batch.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Learning rate step
        scheduler.step()
        
        # Check Early Stopping & Save Best State
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            
        if epoch % 20 == 0 or epoch == 1:
            log_info(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
            
        if epochs_no_improve >= patience:
            log_info(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
            break
            
    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # Save the model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_save_path = os.path.join(MODELS_DIR, "ann_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'seed': seed
    }, model_save_path)
    log_success(f"Best ANN model saved to {model_save_path}")
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2)
    plt.title('Neural Network Training & Validation Loss', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (BCE)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(IMAGES_DIR, exist_ok=True)
    plot_path = os.path.join(IMAGES_DIR, "ann_loss_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    log_success(f"Loss curves plot saved to {plot_path}")

if __name__ == "__main__":
    train_ann()
