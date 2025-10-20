from fastapi import APIRouter, HTTPException
from app.schemas.prediction_schema import LoanApplication
from app.core.logger import logger
import joblib
import pandas as pd
import numpy as np
import os

router = APIRouter()

# Paths to model and preprocessor
MODEL_PATH = "artifacts/model_v1.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor_v1.pkl"

# Load model & preprocessor at startup
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    logger.info("✅ Model and preprocessing pipeline loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading artifacts: {e}")
    model = None
    preprocessor = None

@router.post("/predict")
def predict_loan_status(application: LoanApplication):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not available. Please train the model first.")

    try:
        # Convert input to array in the right order
        input_df = pd.DataFrame([{
    "Income": application.Income,
    "Credit_Score": application.Credit_Score,
    "Loan_Amount": application.Loan_Amount,
    "DTI_Ratio": application.DTI_Ratio,
    "Employment_Status": application.Employment_Status
}])

        logger.info(f"Received input for prediction: {input_df}")

        # Preprocess the data
        processed_data = preprocessor.transform(input_df)
        logger.info("Data preprocessed successfully.")

        # Get prediction
        prediction = model.predict(processed_data)[0]
        result = "Approved" if prediction == 1 else "Rejected"
        logger.info(f"Prediction result: {result}")

        return {"prediction": result}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")
