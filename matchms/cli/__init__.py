"""Command line interface for matchms.

- every command supports ``--help``, ``--json``, ``--quiet`` and ``--verbose``
- data commands emit stable, documented JSON on stdout when a machine
  consumer is detected, or when ``--json`` is passed explicitly
- progress bars and log output go to stderr only
- failures are reported as structured JSON errors with a non-zero exit code
- important results are written to explicit artifacts; stdout only summarizes
  where they were written
"""

from .main import build_parser, main


__all__ = ["build_parser", "main"]
