import logging
from typing import Optional

# Custom verbose level
VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")

def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)

logging.Logger.verbose = verbose

# Create and configure default logger
logger = logging.getLogger("mi_experiments")
logger.setLevel(logging.INFO)

# Add console handler if none exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up a logger with console and optional file handlers."""
    if format_string is None:
        format_string = '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s'
    
    formatter = logging.Formatter(format_string)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Add console handler if none exists
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

__all__ = ['logger', 'setup_logger', 'VERBOSE']
