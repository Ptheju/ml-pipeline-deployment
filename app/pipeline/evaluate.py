import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

from app.core.logger import logger


DATA_PATH = "app//pipeline/data/loan_data.csv"
PREPROCESSOR_PATH = "artifacts/preprocessor_v1.pkl"
MODEL_PATH = "artifacts/model_v1.pkl"
METRICS_PATH = "artifacts/metrics_v1.pkl"

def load_data():
    logger.info("Loading data for evaluaton")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Dataset shape: {df.shape}")
    return df

def preprocess_data(df):
    # Drop unnecessary columns
    if "Text" in df.columns:
        df = df.drop(columns=["Text"])
        logger.info("Dropped 'Text' column")

    # Encode target
    df["Approval"] = df["Approval"].map({"Approved": 1, "Rejected": 0})

    X = df.drop(columns=["Approval"])
    y = df["Approval"]
    return X, y

def evaluate_model():
    logger.info("Evaluating model")

    df = load_data()
    X, y = preprocess_data(df)

    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
        logger.info("Preprocessor and Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading artifacts {e}")

    #preprocess 
    X_preprocessed = preprocessor.transform(X)

    #Split data
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed,y, test_size=0.2, random_state=42)

    #Metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else None

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    # Add AUC if available
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    logger.info(f"Model Evaluation Metrics: {metrics}")

    # Save metrics to JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Metrics saved to: {METRICS_PATH}")

if __name__ == "__main__":
    evaluate_model()

