import os
import sys
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import torch

from utils import MODELS_DIR, log_info, log_success, log_error, log_warning
from train_ann import DiabetesANN
import xgboost as xgb

def get_args():
    parser = argparse.ArgumentParser(description="Predict diabetes risk for a new patient.")
    parser.add_argument("--pregnancies", type=float, help="Number of pregnancies")
    parser.add_argument("--glucose", type=float, help="Plasma glucose concentration")
    parser.add_argument("--bp", type=float, help="Diastolic blood pressure (mm Hg)")
    parser.add_argument("--thickness", type=float, help="Triceps skin fold thickness (mm)")
    parser.add_argument("--insulin", type=float, help="2-Hour serum insulin (mu U/ml)")
    parser.add_argument("--bmi", type=float, help="Body mass index (weight in kg/(height in m)^2)")
    parser.add_argument("--pedigree", type=float, help="Diabetes pedigree function score")
    parser.add_argument("--age", type=float, help="Age in years")
    return parser.parse_args()

def interactive_input():
    print("\n--- Enter Patient Medical Metrics ---")
    inputs = {}
    
    # Prompt helper
    def prompt_float(field_name, description, allow_zero_as_missing=True):
        suffix = " (Enter 0 if missing/unknown)" if allow_zero_as_missing else ""
        while True:
            try:
                val = input(f"{description}{suffix}: ").strip()
                val_float = float(val)
                if val_float < 0:
                    print("Value cannot be negative. Please enter a valid number.")
                    continue
                return val_float
            except ValueError:
                print("Invalid input. Please enter a number.")

    inputs["Pregnancies"] = prompt_float("Pregnancies", "Number of Pregnancies", allow_zero_as_missing=False)
    inputs["Glucose"] = prompt_float("Glucose", "Glucose level")
    inputs["BloodPressure"] = prompt_float("BloodPressure", "Blood Pressure")
    inputs["SkinThickness"] = prompt_float("SkinThickness", "Skin Thickness")
    inputs["Insulin"] = prompt_float("Insulin", "Insulin level")
    inputs["BMI"] = prompt_float("BMI", "BMI value")
    inputs["DiabetesPedigreeFunction"] = prompt_float("DiabetesPedigreeFunction", "Diabetes Pedigree Score", allow_zero_as_missing=False)
    inputs["Age"] = prompt_float("Age", "Age in years", allow_zero_as_missing=False)
    
    return inputs

def load_preprocessors():
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    medians_path = os.path.join(MODELS_DIR, "medians.json")
    
    if not os.path.exists(scaler_path) or not os.path.exists(medians_path):
        raise FileNotFoundError("Preprocessing artifacts not found in models/ directory. Please run data preprocessing first.")
        
    scaler = joblib.load(scaler_path)
    with open(medians_path, "r") as f:
        medians = json.load(f)
        
    return scaler, medians

def load_models(device):
    xgb_path = os.path.join(MODELS_DIR, "xgboost_model.json")
    ann_path = os.path.join(MODELS_DIR, "ann_model.pth")
    
    if not os.path.exists(xgb_path) or not os.path.exists(ann_path):
        raise FileNotFoundError("Trained models not found. Please run the training scripts first.")
        
    # Load XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)
    
    # Load ANN
    checkpoint = torch.load(ann_path, map_location=device)
    ann_model = DiabetesANN(checkpoint['input_dim'])
    ann_model.load_state_dict(checkpoint['model_state_dict'])
    ann_model.to(device)
    ann_model.eval()
    
    return xgb_model, ann_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load preprocessors and models
    try:
        scaler, medians = load_preprocessors()
        xgb_model, ann_model = load_models(device)
    except FileNotFoundError as e:
        log_error(str(e))
        sys.exit(1)
        
    args = get_args()
    
    # Check if CLI args are provided, else fallback to interactive
    cli_provided = any(v is not None for v in vars(args).values())
    if cli_provided:
        # Check if all arguments are provided
        missing_args = [k for k, v in vars(args).items() if v is None]
        if missing_args:
            log_error(f"Incomplete arguments. Missing: {missing_args}")
            log_info("Please provide all CLI options or run the script without any arguments for interactive mode.")
            sys.exit(1)
            
        patient_data = {
            "Pregnancies": args.pregnancies,
            "Glucose": args.glucose,
            "BloodPressure": args.bp,
            "SkinThickness": args.thickness,
            "Insulin": args.insulin,
            "BMI": args.bmi,
            "DiabetesPedigreeFunction": args.pedigree,
            "Age": args.age
        }
    else:
        patient_data = interactive_input()
        
    log_info("Applying preprocessing to new patient data...")
    
    # Create copy for modification
    processed_data = patient_data.copy()
    
    # 1. Impute missing zero values with training set medians
    impute_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in impute_columns:
        if processed_data[col] == 0:
            median_val = medians[col]
            processed_data[col] = median_val
            log_warning(f"Feature '{col}' entered as 0 (missing). Automatically imputed with training median: {median_val}")
            
    # 2. Scale features using the StandardScaler
    features_in_order = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    input_vector = [processed_data[f] for f in features_in_order]
    
    # Format to DataFrame to prevent scaling warnings regarding feature names
    input_df = pd.DataFrame([input_vector], columns=features_in_order)
    input_scaled = scaler.transform(input_df)
    
    # 3. Model Inference
    # XGBoost
    xgb_prob = xgb_model.predict_proba(input_scaled)[0, 1]
    xgb_pred = int(xgb_prob >= 0.5)
    
    # PyTorch ANN
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        ann_logits = ann_model(input_tensor)
        ann_prob = torch.sigmoid(ann_logits).cpu().item()
        ann_pred = int(ann_prob >= 0.5)
        
    # Print Results
    print("\n" + "=" * 50)
    print("           DIABETES RISK PREDICTION RESULTS           ")
    print("=" * 50)
    
    print("\nPatient Metrics Summary:")
    for f in features_in_order:
        orig = patient_data[f]
        proc = processed_data[f]
        status = " (Imputed)" if orig != proc else ""
        print(f"  - {f:25}: {orig:<8} -> Scaled: {input_scaled[0, features_in_order.index(f)]:.4f}{status}")
        
    print("\nModel Risk Assessments:")
    
    # Helper to style risk rating
    def risk_level(prob):
        if prob < 0.3:
            return "Low Risk"
        elif prob < 0.7:
            return "Moderate Risk"
        else:
            return "High Risk"
            
    print(f"  [XGBoost Classifier]")
    print(f"    - Probability of Diabetes: {xgb_prob:.2%}")
    print(f"    - Decision Threshold:      {risk_level(xgb_prob)}")
    print(f"    - Prediction Outcome:      {'DIABETIC (Positive)' if xgb_pred == 1 else 'NON-DIABETIC (Negative)'}")
    
    print(f"\n  [PyTorch Neural Network]")
    print(f"    - Probability of Diabetes: {ann_prob:.2%}")
    print(f"    - Decision Threshold:      {risk_level(ann_prob)}")
    print(f"    - Prediction Outcome:      {'DIABETIC (Positive)' if ann_pred == 1 else 'NON-DIABETIC (Negative)'}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
