import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from app.core.logger import logger

def preprocess_input(data:dict, scaler):

    df = pd.DataFrame([data])

   

    processed_data = scaler.transform(df)

     # Debug print
    logger.info(f"Raw input: {df.to_dict(orient='records')}")
    logger.info(f"Scaled data: {processed_data.tolist()}")

    return processed_data
