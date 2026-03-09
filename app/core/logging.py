import sys
from loguru import logger
import os

def setup_logging():
    # Remove existing handlers
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG",
    )
    
    # Add file handler
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logger.add(
        os.path.join(log_dir, "app.log"),
        rotation="10 MB",
        retention="10 days",
        level="INFO",
        compression="zip"
    )

setup_logging()
