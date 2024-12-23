from dataclasses import dataclass
import logging

@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    level: int = logging.INFO
    log_to_file: bool = True
    log_to_console: bool = True
    format: str = '%(asctime)s - %(levelname)s - %(message)s'