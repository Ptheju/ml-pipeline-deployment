import joblib 
import os
from app.core.logger import logger
from app.core.config import MODEL_PATH, SCALER_PATH


def load_model_and_scaler():
    logger.info(f"loading model from:{MODEL_PATH}")
    logger.info(f"loading scaler from:{SCALER_PATH}")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("model and scaler loaded successfully")
    return model, scaler 
