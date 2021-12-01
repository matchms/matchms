"""Matchms logger.

Matchms functions and method report unexpected or undesired behavior as
logging WARNING, and additional information as INFO.
The default logging level is set to WARNING. If you want to output additional
logging messages, you can lower the logging level to INFO using setLevel:

.. code-block:: python

    from matchms import set_matchms_logger_level
    from matchms import calculate_scores, Spectrum


    set_matchms_logger_level("INFO")

If you want to suppress logging warnings, you can also raise the logging level
to ERROR by:

.. code-block:: python

    set_matchms_logger_level("ERROR")

To write logging entries to a local file, you can do the following:

.. code-block:: python

    from matchms import add_logging_to_file
    from matchms import calculate_scores, Spectrum

    add_logging_to_file("sample.log")

"""
import logging
import logging.config
import sys


_formatter = logging.Formatter(
    '%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(message)s'
    )


def _init_logger():
    """Initialize matchms logger."""
    logger = logging.getLogger('matchms')
    logger.setLevel(logging.WARNING)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(_formatter)
    logger.addHandler(handler)
    logger.info('Completed configuring matchms logger.')


def set_matchms_logger_level(loglevel: str):
    """Update logging level to given loglevel.

    Parameters
    ----------
    loglevels
        Can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    """
    level = logging.getLevelName(loglevel)
    logger = logging.getLogger("matchms")
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def add_logging_to_file(filename: str, loglevel: str = "INFO",
                        remove_stream_handlers: bool = False):
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
    """
    level = logging.getLevelName(loglevel)
    logger = logging.getLogger("matchms")
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(_formatter)
    logger.addHandler(file_handler)
    # Remove the existing handlers if they are not of type FileHandler
    if remove_stream_handlers is True:
        for handler in logger.handlers:
            if not isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)


def reset_matchms_logger():
    logger = logging.getLogger("matchms")
    logger.handlers.clear()
    _init_logger()
