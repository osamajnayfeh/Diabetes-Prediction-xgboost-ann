import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

from utils import DATA_DIR, MODELS_DIR, IMAGES_DIR, log_info, log_success, set_seed

def train_xgboost(seed=42):
    set_seed(seed)
    
    # Load dataset splits
    log_info("Loading preprocessed splits for XGBoost...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    
    # Combine train and validation for grid search cross-validation
    # This maximizes the training data while retaining the test split strictly for evaluation
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    
    X_train = full_train_df.drop(columns=["Outcome"])
    y_train = full_train_df["Outcome"]
    
    log_info(f"Combined Train+Val shape for cross-validation: {X_train.shape}")
    
    # Set up hyperparameter grid
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 150, 200],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
    
    log_info("Initializing XGBClassifier...")
    estimator = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=seed
    )
    
    log_info("Running GridSearchCV (5-fold CV)...")
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    log_success(f"Grid search complete. Best ROC-AUC: {best_score:.4f}")
    log_info(f"Best hyperparameters: {best_params}")
    
    # Train final model with best parameters
    best_model = grid_search.best_estimator_
    
    # Save the model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "xgboost_model.json")
    best_model.save_model(model_path)
    log_success(f"Best XGBoost model saved to {model_path}")
    
    # Plot feature importance
    log_info("Plotting feature importances...")
    feature_importances = best_model.feature_importances_
    features = X_train.columns
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Set premium aesthetic styling
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Premium gradient color palette using HSL-like tailored blues
    colors = sns.color_palette("Blues_r", n_colors=len(features))
    
    sns.barplot(
        x='Importance',
        y='Feature',
        data=importance_df,
        palette=colors,
        hue='Feature',
        legend=False
    )
    
    plt.title('XGBoost Feature Importance for Diabetes Detection', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    
    plot_path = os.path.join(IMAGES_DIR, "xgboost_feature_importance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    log_success(f"Feature importance plot saved to {plot_path}")
    
if __name__ == "__main__":
    train_xgboost()
