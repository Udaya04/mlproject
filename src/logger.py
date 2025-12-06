import logging
import os
from datetime import datetime

# Create logs directory correctly
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Generate clean log file path
log_filename = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, log_filename)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='[%(asctime)s] %(lineno)d %(name)s %(levelname)s - %(message)s',
)

if __name__ == "__main__":
    logging.info("Logging setup complete.")

    