import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from app.core.logger import logger

def build_preprocessing_pipeline(df: pd.DataFrame):

    #Detect column types

    numeric_features = df.select_dtypes(include=['int64','float64']).columns.tolist()
    categoric_features = df.select_dtypes(include=['object','category']).columns.tolist()

    logger.info(f"Detected numeric features:{numeric_features}")
    logger.info(f"Detected categoric features:{categoric_features}")

    #Numeric Data preprocessing steps

    numeric_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler',StandardScaler())])

    #Categorical preprocessing

    categoric_pipeline = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
    
    #Combine both pipelines

    preprocessor = ColumnTransformer(transformers=[('num',numeric_pipeline,numeric_features),('cat',categoric_pipeline,categoric_features)])

    logger.info("preprocessing piplines successfully created")

    return preprocessor, numeric_features, categoric_features
