import os
import pandas as pd
import numpy as np
import torch
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from tabulate import tabulate

from utils import DATA_DIR, MODELS_DIR, IMAGES_DIR, RESULTS_DIR, log_info, log_success, log_error
from train_ann import DiabetesANN

def load_xgboost_model():
    """Load the trained XGBoost model."""
    model_path = os.path.join(MODELS_DIR, "xgboost_model.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"XGBoost model not found at {model_path}")
    
    # Load model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

def load_ann_model(device):
    """Load the trained PyTorch ANN model."""
    model_path = os.path.join(MODELS_DIR, "ann_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ANN model not found at {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    input_dim = checkpoint['input_dim']
    
    model = DiabetesANN(input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def evaluate_models():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test split
    test_path = os.path.join(DATA_DIR, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test split not found at {test_path}. Please run data preprocessing first.")
        
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["Outcome"])
    y_test = test_df["Outcome"].values
    
    log_info("Loading models for evaluation...")
    xgb_model = load_xgboost_model()
    ann_model = load_ann_model(device)
    
    # ------------------
    # XGBoost Predictions
    # ------------------
    log_info("Running XGBoost inference...")
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # ------------------
    # ANN Predictions
    # ------------------
    log_info("Running ANN inference...")
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        ann_logits = ann_model(X_test_tensor)
        ann_probs = torch.sigmoid(ann_logits).cpu().numpy().flatten()
        ann_preds = (ann_probs >= 0.5).astype(int)
        
    # Calculate Metrics
    metrics = {}
    for name, y_pred, y_prob in [("XGBoost", xgb_preds, xgb_probs), ("Neural Network", ann_preds, ann_probs)]:
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        }
        
    # Prepare comparison table
    table_data = []
    for model_name, score_dict in metrics.items():
        row = [model_name] + [f"{score_dict[m]*100:.2f}%" if m != "ROC-AUC" else f"{score_dict[m]:.4f}" for m in score_dict]
        table_data.append(row)
        
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    markdown_table = tabulate(table_data, headers=headers, tablefmt="github")
    
    print("\n--- Model Performance Comparison (Test Set) ---")
    print(markdown_table)
    print("------------------------------------------------\n")
    
    # Save results to a file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.md")
    with open(results_path, "w") as f:
        f.write("# Model Evaluation Results\n\n")
        f.write("This file contains the final comparison metrics between XGBoost and the Artificial Neural Network on the holdout test set (15% of the dataset).\n\n")
        f.write("## Test Set Performance Comparison\n\n")
        f.write(markdown_table)
        f.write("\n\n## Analysis Summary\n")
        
        # Simple heuristic to write a dynamic analysis summary
        if metrics["XGBoost"]["ROC-AUC"] > metrics["Neural Network"]["ROC-AUC"]:
            better_model = "XGBoost"
            other_model = "Neural Network"
        else:
            better_model = "Neural Network"
            other_model = "XGBoost"
            
        f.write(f"- **Top Performer:** `{better_model}` achieved a higher ROC-AUC of `{metrics[better_model]['ROC-AUC']:.4f}` compared to `{metrics[other_model]['ROC-AUC']:.4f}` for `{other_model}`.\n")
        f.write(f"- **Clinical Context:** For diabetes screening, high **Recall (Sensitivity)** is often critical to ensure patients with diabetes are not missed. ")
        f.write(f"`XGBoost` achieved a recall of `{metrics['XGBoost']['Recall']*100:.2f}%`, while the `Neural Network` achieved `{metrics['Neural Network']['Recall']*100:.2f}%`.\n")
        f.write(f"- **Model Size & Complexity:** XGBoost is highly optimized for structured tabular data of this scale, while the Neural Network has structural regularizations (Dropout/BatchNorm) to counter overfitting on this relatively small dataset of 768 samples.\n")
        
    log_success(f"Evaluation report saved to {results_path}")
    
    # ------------------
    # Plotting ROC Curves
    # ------------------
    log_info("Plotting ROC Curves...")
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    for name, y_prob, color in [("XGBoost", xgb_probs, "#1f77b4"), ("Neural Network", ann_probs, "#ff7f0e")]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = metrics[name]["ROC-AUC"]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", color=color, linewidth=2)
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11)
    
    os.makedirs(IMAGES_DIR, exist_ok=True)
    roc_plot_path = os.path.join(IMAGES_DIR, "roc_curve_comparison.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    log_success(f"ROC comparison curve saved to {roc_plot_path}")
    
    # ------------------
    # Plotting Confusion Matrices
    # ------------------
    log_info("Plotting Confusion Matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, (name, y_pred, cmap) in enumerate([("XGBoost", xgb_preds, "Blues"), ("Neural Network", ann_preds, "Oranges")]):
        cm = confusion_matrix(y_test, y_pred)
        # Format labels: True Neg, False Pos, False Neg, True Pos
        group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
        
        labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        
        sns.heatmap(
            cm, annot=labels, fmt="", cmap=cmap, cbar=False, ax=axes[i],
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes']
        )
        axes[i].set_title(f'{name} Confusion Matrix', fontsize=14, fontweight='bold', pad=10)
        axes[i].set_xlabel('Predicted Label', fontsize=12)
        axes[i].set_ylabel('True Label', fontsize=12)
        
    plt.tight_layout()
    cm_plot_path = os.path.join(IMAGES_DIR, "confusion_matrices_comparison.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    log_success(f"Confusion matrices comparison saved to {cm_plot_path}")

if __name__ == "__main__":
    evaluate_models()
