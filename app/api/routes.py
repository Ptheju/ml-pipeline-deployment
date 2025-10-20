from fastapi import APIRouter
from app.schemas.prediction_schema import PredictionRequest, PredictionResponse
from app.models.model_loader import load_model_and_scaler
from app.pipeline.preprocess import preprocess_input

router = APIRouter()

model, scaler = load_model_and_scaler()

@router.post("/predict",response_model=PredictionResponse)

def predict(request:PredictionRequest):
    data = request.dict()

    processed_data = preprocess_input(data, scaler)
    prediction = model.predict(processed_data)[0]
    print("Processed input:", processed_data)
    return {"prediction": float(prediction)}