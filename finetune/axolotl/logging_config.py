"""
Common logging module for axolotl
"""

import os
import sys
from logging import Formatter
from logging.config import dictConfig
from typing import Any, Dict

from colorama import Fore, Style, init
from tqdm.contrib.logging import logging_redirect_tqdm

class ColorfulFormatter(Formatter):
    """
    Formatter to add coloring to log messages by log type
    """

    COLORS = {
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
        "INFO": Fore.GREEN,
        "DEBUG": Fore.BLUE
    }

    def format(self, record):
        record.rank = int(os.getenv("LOCAL_RANK", "0"))
        log_message = super().format(record)
        return self.COLORS.get(record.levelname, "") + log_message + Fore.RESET


DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d] %(message)s",
        },
        "file_simple": {
            "format": "%(asctime)s | %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        },
        "colorful": {
            "()": ColorfulFormatter,
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d] [RANK:%(rank)d] %(message)s",
        },
    },
    "filters": {},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "filters": [],
            "stream": sys.stdout,
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "file_simple",  # Choose the appropriate formatter
            "filename": os.getenv("LOG_FILE", "axolotl.log"),  # Specify the file path
            "mode": "a",
        },
        "color_console": {
            "class": "logging.StreamHandler",
            "formatter": "colorful",
            "filters": [],
            "stream": sys.stdout,
        },
    },
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO") },
    "loggers": {
        "axolotl": {
            "handlers": ["color_console", "file"],
            "level": os.getenv("LOG_LEVEL", "DEBUG") if int(os.getenv("LOCAL_RANK", "0")) in [-1,0] else "WARNING",
            "propagate": False,
        },
        "transformers": {
            # default is warning
            "handlers": ["file"],
            "level": os.getenv("LOG_LEVEL", "INFO") if int(os.getenv("LOCAL_RANK", "0")) in [-1,0] else "WARNING",
            "propagate": False,   
        }
    },
}


def configure_logging():
    """Configure with default logging"""
    if os.getenv("LOG_FILE", False):
        os.makedirs(os.path.dirname(os.getenv("LOG_FILE")), exist_ok=True)
    init()  # Initialize colorama
    dictConfig(DEFAULT_LOGGING_CONFIG)
