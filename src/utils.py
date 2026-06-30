import os
import random
import numpy as np
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Raw data file path
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_diabetes.csv")

def set_seed(seed=42):
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_info(msg):
    """Print an informational log message."""
    print(f"\033[94m[INFO]\033[0m {msg}")

def log_success(msg):
    """Print a success log message."""
    print(f"\033[92m[SUCCESS]\033[0m {msg}")

def log_warning(msg):
    """Print a warning log message."""
    print(f"\033[93m[WARNING]\033[0m {msg}")

def log_error(msg):
    """Print an error log message."""
    print(f"\033[91m[ERROR]\033[0m {msg}")
