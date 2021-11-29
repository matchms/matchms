# -*- coding: utf-8 -*-
import logging
from matchms.logging import logger


def test_logger_settings(caplog):
    """Test logging functionality."""
    logger.info("info test")
    logger.warning("warning test")
    assert logger.name == "matchms", "Expected different logger name"
    assert logger.getEffectiveLevel() == 30, "Expected different logging level"
    assert "info test" not in caplog.text, "Info log should not be shown."
    assert "warning test" in caplog.text, "Warning log should have been shown."


def test_logger_change_of_output_level(caplog):
    """Test logging functionality."""
    logging.getLogger("matchms").setLevel(logging.INFO)
    logger.debug("debug test")
    logger.info("info test")

    assert logger.name == "matchms", "Expected different logger name"
    assert logger.getEffectiveLevel() == 20, "Expected different logging level"
    assert "debug test" not in caplog.text, "Debug log should not be shown."
    assert "info test" in caplog.text, "Info log should have been shown."
