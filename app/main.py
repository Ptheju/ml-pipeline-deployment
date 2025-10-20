from fastapi import FastAPI
from app.api.predict_route import router as predict_router
from app.core.logger import logger
from app.core.config import API_TITLE, API_DESCRIPTION, API_VERSION, ENVIRONMENT

app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version= API_VERSION)

# Log startup information
logger.info(f"Starting application in {ENVIRONMENT} mode...")

# Register the prediction route
app.include_router(predict_router, prefix="/api")

@app.get("/")

def root():
    logger.info("Root endpoint accessed")
    return {"Welcome to ML Pipeline Deployment"}