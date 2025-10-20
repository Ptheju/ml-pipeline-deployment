from fastapi import FastAPI
from app.api.routes import router
from app.core.logger import logger
from app.core.config import API_TITLE, API_DESCRIPTION, API_VERSION, ENVIRONMENT

app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version= API_VERSION)

# Log startup information
logger.info(f"Starting application in {ENVIRONMENT} mode...")

app.include_router(router, prefix="/api", tags=["prediction"])

@app.get("/")

def root():
    logger.info("Root endpoint accessed")
    return {"Welcome to ML Pipeline Deployment"}