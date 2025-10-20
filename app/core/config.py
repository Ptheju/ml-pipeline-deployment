import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Model file paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")

# API Metadata
API_TITLE = "ML Pipeline Deployment API"
API_DESCRIPTION = "End-to-End ML pipeline with preprocessing and model prediction"
API_VERSION = "1.0.0"

# Environment (future proofing)
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
