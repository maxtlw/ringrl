import logging
import sys


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Don't double-log via root handlers if user configured global logging too.
    logger.propagate = False

    # Add a single stream handler (or update existing ones) so formatting is stable.
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if logger.handlers:
        for h in logger.handlers:
            h.setLevel(level)
            if h.formatter is None:
                h.setFormatter(fmt)
        return logger

    # Use stdout so it shows up in terminals/debug consoles more consistently.
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger
