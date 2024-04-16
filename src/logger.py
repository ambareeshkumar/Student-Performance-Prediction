import logging
import os
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(),"logs",LOG_FILE)

os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)

class ColoredFormatter(logging.Formatter):
    COLOR_CODES = {
        logging.DEBUG: '\033[0m',  # Default color
        logging.INFO: '\033[92m',   # Green
        logging.WARNING: '\033[93m',  # Yellow
        logging.ERROR: '\033[91m',    # Red
        logging.CRITICAL: '\033[91m'  # Red
    }

    def format(self, record):
        log_color = self.COLOR_CODES.get(record.levelno, self.COLOR_CODES[logging.DEBUG])
        log_msg = super().format(record)
        return f"{log_color}{log_msg}\033[0m"

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)


# Create a console handler for logging to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Set the formatter for the console handler
console_formatter = ColoredFormatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# Add the console handler to the root logger
logging.getLogger().addHandler(console_handler)