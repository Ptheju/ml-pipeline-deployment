from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from app.pipeline.preprocess import build_preprocessing_pipeline
from app.core.logger import logger


import joblib


#def generate_synthetic_data(n=500):
#    np.random.seed(42)
#   X = np.random.rand(n,3)
#    y = (X.sum(axis=1)>1.5).astype(int)
#   return X, y

DATA_PATH = "app/pipeline/data/loan_data.csv"
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model_v1.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor_v1.pkl")


def load_data():
    logger.info("loading datasets..")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Dataset loaded with shape:{df.shape}")
    return df

def preprocess_data(df):
    #Drop "Text" Column
    if "Text" in df.columns:
        df = df.drop(columns=["Text"])
        logger.info("Dropped 'Text' Column from Dataset")
    
    if "Approval" not in df.columns:
        raise ValueError("Target Column 'Approval' not found in datasets")
    
    df["Approval"] = df["Approval"].map({"Approved":1, "Rejected":0})
    logger.info("Encoded target column Approval")

    X = df.drop(columns=["Approval"])
    y = df["Approval"]

    return X, y

def train_model(X_train,y_train):
    logger.info("Training Random Forest Model.. ")
    model = RandomForestClassifier(n_estimators=200,random_state=42)
    model.fit(X_train,y_train)
    logger.info("Model training completed")
    return model

def save_artifacts(model,preprocessor):
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
    
    joblib.dump(model,MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info(f"Artifacts saved: Model -> {MODEL_PATH}, Preprocessor -> {PREPROCESSOR_PATH}")

def main():
    logger.info("Started training pipeline")
    
    #Load Data
    df = load_data()

    #Preprocess Data
    X, y = preprocess_data(df)

    #Build preprocessing pipeline
    preprocessor, numeric_features, categoric_features = build_preprocessing_pipeline(X)
    logger.info("Fitting preprocessing pipeline...")
    X_preprocessed = preprocessor.fit_transform(X)


    #Split Data

    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed,y,test_size=0.2,random_state=42)
    logger.info("Data split into train and test sets")

    #Train model

    model = train_model(X_train,y_train)

    #Save artifacts

    save_artifacts(model,preprocessor)

    logger.info("Training pipeline finished successfully")




if __name__=='__main__':
    main()

