"""
Chunker Logging Module
Replaces print() statements with structured logging
"""

import logging
import os
from .config import LOGGING, PATHS

# Create logs directory
os.makedirs(PATHS['logs_dir'], exist_ok=True)

# Create logger instance
logger = logging.getLogger('chunker')
logger.setLevel(getattr(logging, LOGGING['level']))

# Remove default handlers
logger.handlers = []

# Create formatters
formatter = logging.Formatter(
    LOGGING['format'],
    datefmt=LOGGING['date_format']
)

# Console handler (always output to console)
console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, LOGGING['level']))
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler (if enabled)
if LOGGING['log_to_file']:
    log_file = os.path.join(PATHS['logs_dir'], LOGGING['log_file'].split('/')[-1])
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(getattr(logging, LOGGING['level']))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Convenience functions
def info(msg: str, *args, **kwargs):
    """Log info level message"""
    logger.info(msg, *args, **kwargs)

def warning(msg: str, *args, **kwargs):
    """Log warning level message"""
    logger.warning(msg, *args, **kwargs)

def error(msg: str, *args, **kwargs):
    """Log error level message"""
    logger.error(msg, *args, **kwargs)

def debug(msg: str, *args, **kwargs):
    """Log debug level message"""
    logger.debug(msg, *args, **kwargs)

def critical(msg: str, *args, **kwargs):
    """Log critical level message"""
    logger.critical(msg, *args, **kwargs)

# Export logger instance for direct use
__all__ = ['logger', 'info', 'warning', 'error', 'debug', 'critical']
