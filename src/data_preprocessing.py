import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import DATA_DIR, RAW_DATA_PATH, log_info, log_success, log_error, set_seed

DATASET_URL = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"

def download_dataset():
    """Download the diabetes dataset if it doesn't exist locally."""
    if not os.path.exists(RAW_DATA_PATH):
        log_info(f"Downloading dataset from {DATASET_URL}...")
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            df = pd.read_csv(DATASET_URL)
            df.to_csv(RAW_DATA_PATH, index=False)
            log_success(f"Dataset downloaded successfully and saved to {RAW_DATA_PATH}")
        except Exception as e:
            log_error(f"Failed to download dataset: {e}")
            raise e
    else:
        log_info(f"Raw dataset already exists at {RAW_DATA_PATH}")

def preprocess_and_split(seed=42):
    """
    Load raw data, split into train/val/test, impute missing values (zeros),
    scale features, and save preprocessed splits.
    """
    set_seed(seed)
    
    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)
    log_info(f"Loaded raw dataset with shape: {df.shape}")
    
    # Identify features and target
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    
    # Train/Val/Test Split (70% Train, 15% Val, 15% Test)
    # Stratify split by the outcome to maintain class balance
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    
    log_info(f"Data splits: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    
    # Impute missing values (0 values in specific columns)
    # Zeros in these columns represent missing values, not real measurements
    cols_to_impute = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    
    # We must calculate medians on the TRAINING set only to prevent data leakage!
    medians = {}
    for col in cols_to_impute:
        # Calculate median of non-zero elements in training set
        train_median = X_train[X_train[col] != 0][col].median()
        medians[col] = train_median
        
        # Apply imputation to all splits
        X_train[col] = X_train[col].replace(0, train_median)
        X_val[col] = X_val[col].replace(0, train_median)
        X_test[col] = X_test[col].replace(0, train_median)
        
    log_info(f"Imputed columns {cols_to_impute} with train medians: {medians}")
    
    # Scale features
    scaler = StandardScaler()
    
    # Fit on training data and transform all splits
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    feature_names = X.columns.tolist()
    
    # Reconstruct dataframes with Outcome column
    train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    train_df["Outcome"] = y_train.values
    
    val_df = pd.DataFrame(X_val_scaled, columns=feature_names)
    val_df["Outcome"] = y_val.values
    
    test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    test_df["Outcome"] = y_test.values
    
    # Save datasets
    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    log_success("Preprocessed splits saved successfully to the data directory.")
    return train_df, val_df, test_df

if __name__ == "__main__":
    download_dataset()
    preprocess_and_split()
