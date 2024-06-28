"""Matchms logger.

Matchms functions and method report unexpected or undesired behavior as
logging WARNING, and additional information as INFO.
The default logging level is set to WARNING. If you want to output additional
logging messages, you can lower the logging level to INFO using set_matchms_logger_level:

.. code-block:: python

    from matchms import set_matchms_logger_level

    set_matchms_logger_level("INFO")

If you want to suppress logging warnings, you can also raise the logging level
to ERROR by:

.. code-block:: python

    set_matchms_logger_level("ERROR")

To write logging entries to a local file, you can do the following:

.. code-block:: python

    from matchms.logging_functions import add_logging_to_file

    add_logging_to_file("sample.log", loglevel="INFO")

If you want to write the logging messages to a local file while silencing the
stream of such messages, you can do the following:

.. code-block:: python

    from matchms.logging_functions import add_logging_to_file

    add_logging_to_file("sample.log", loglevel="INFO",
                        remove_stream_handlers=True)

"""
import logging
import logging.config
import sys


try:
    from rdkit import RDLogger
except ImportError:
    _has_rdkit = False
else:
    _has_rdkit = True
rdkit_missing_message = "Conda package 'rdkit' is required for this functionality."


_formatter = logging.Formatter(
    '%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(message)s')


def _init_logger(logger_name="matchms"):
    """Initialize matchms logger."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(_formatter)
    logger.addHandler(handler)
    logger.info('Completed configuring matchms logger.')


def set_matchms_logger_level(loglevel: str, logger_name="matchms"):
    """Update logging level to given loglevel.

    Parameters
    ----------
    loglevels
        Can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    logger_name
        Default is "matchms". Change if logger name should be different.
    """
    level = logging.getLevelName(loglevel)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def add_logging_to_file(filename: str, loglevel: str = "INFO",
                        remove_stream_handlers: bool = False,
                        logger_name="matchms"):
    """Add logging to file.

    Current implementation does not change the initial logging stream,
    but simply adds a FileHandler to write logging entries to a file.

    Parameters
    ----------
    filename
        Name of file for write logging output.
    loglevels
        Can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    remove_stream_handlers
        Set to True if only logging to file is desired.
    logger_name
        Default is "matchms". Change if logger name should be different.
    """
    level = logging.getLevelName(loglevel)
    logger = logging.getLogger(logger_name)
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(_formatter)
    logger.addHandler(file_handler)

    # Remove the existing handlers if they are not of type FileHandler
    if remove_stream_handlers is True:
        for handler in logger.handlers:
            if not isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)


def reset_matchms_logger(logger_name="matchms"):
    """Reset matchms logger to initial state.

    This will remove all logging Handlers and initialize a new matchms logger.
    Use this function to reset previous changes made to the default matchms logger.

    Parameters
    ----------
    logger_name
        Default is "matchms". Change if logger name should be different.
    """
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    _init_logger()


def set_rdkit_logger_level(level):
    """Rdkit log levels severity order.

    This function recreates the functionality of the rdkit function.
    Created based on https://github.com/rdkit/rdkit/blob/master/rdkit/RDLogger.py setLevel()

    Parameters
    ----------
    level
        Refers to rdkit log levels ('rdApp.debug', 'rdApp.info', 'rdApp.warning', 'rdApp.error').
    """
    rdkit_log_levels = ['rdApp.debug', 'rdApp.info', 'rdApp.warning', 'rdApp.error']
    
    if not _has_rdkit:
        raise ImportError(rdkit_missing_message)

    if level not in rdkit_log_levels:
        raise ValueError(f"Invalid log level. Allowed values are: {rdkit_log_levels}")
    # convert string level to index
    level_index = rdkit_log_levels.index(level)

    # enable all levels with higher severity
    for i in range(level_index, len(rdkit_log_levels)):
        RDLogger.EnableLog(rdkit_log_levels[i])

    # disable all levels with lower severity
    for i in range(0, level_index):
        RDLogger.DisableLog(rdkit_log_levels[i])
