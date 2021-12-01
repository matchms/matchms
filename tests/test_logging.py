# -*- coding: utf-8 -*-
import logging
from matchms.logging_functions import set_matchms_logger_level


def test_initial_logging(caplog):
    """Test logging functionality."""
    logger = logging.getLogger("matchms")
    logger.info("info test")
    logger.warning("warning test")
    assert logger.name == "matchms", "Expected different logger name"
    assert logger.getEffectiveLevel() == 30, "Expected different logging level"
    assert "info test" not in caplog.text, "Info log should not be shown."
    assert "warning test" in caplog.text, "Warning log should have been shown."


def test_set_matchms_logger_level(caplog):
    """Test logging functionality."""
    logger = logging.getLogger("matchms")
    set_matchms_logger_level("INFO")
    logger.debug("debug test")
    logger.info("info test")

    assert logger.name == "matchms", "Expected different logger name"
    assert logger.getEffectiveLevel() == 20, "Expected different logging level"
    assert "debug test" not in caplog.text, "Debug log should not be shown."
    assert "info test" in caplog.text, "Info log should have been shown."
