from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib


#def generate_synthetic_data(n=500):
#    np.random.seed(42)
#   X = np.random.rand(n,3)
#    y = (X.sum(axis=1)>1.5).astype(int)
#   return X, y



def train_and_save_model():
    df = pd.read_csv("data.csv")
    

    X = df[["feature1","feature2","feature3"]]
    y = df["target"]

    #create and fit scalar

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Train model

    model = LogisticRegression()
    model.fit(X_scaled,y)

    #Save model and scaler

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    joblib.dump(model, os.path.join(models_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    print("Model and Scaler Saved Successfully!")


if __name__=='__main__':
    train_and_save_model()

