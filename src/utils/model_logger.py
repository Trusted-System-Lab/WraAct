"""
This module provides the logger builder for the experiment.

.. tip::
    There will be a global logger named "rover" for the experiment.
"""

__docformat__ = "restructuredtext"
__all__ = ["build_logger"]

import logging
import sys

from .model_args import LoggerArgs


def build_logger(logger_args: LoggerArgs, name: str = None) -> logging.Logger:
    """
    Build a logger with the given arguments.

    :param logger_args: The arguments for the logger.
    :param name: The name of the logger.
    """
    logger_ = logging.getLogger(name)
    logger_.setLevel(logger_args.log_level)

    # Create the console handler
    if logger_args.log_console:
        ch = logging.StreamHandler()
        ch.setLevel(logger_args.log_level)
        ch.setFormatter(logging.Formatter(logger_args.log_format))
        logger_.addHandler(ch)

    # Create the file handler
    if logger_args.log_file:
        fh = logging.FileHandler(logger_args.log_file, mode="w")
        fh.setLevel(logger_args.log_level)
        fh.setFormatter(logging.Formatter(logger_args.log_format))
        logger_.addHandler(fh)

    def _handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger_.error("Exception: ", exc_info=(exc_type, exc_value, exc_traceback))

    if logger_args.log_console or logger_args.log_file:
        sys.excepthook = _handle_exception

    return logger_
