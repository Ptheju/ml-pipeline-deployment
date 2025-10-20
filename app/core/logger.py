import logging

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# Logger instance
logger = logging.getLogger("ml_pipeline_app")
