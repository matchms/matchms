"""Structured error handling for the matchms CLI.

Every failure is turned into a ``CliError`` with a machine-readable code, the
offending input/parameter, the valid options (when applicable) and a flag that
says whether any output artifact was produced. ``main`` renders these as a
JSON document on stdout and exits with a non-zero code — no tracebacks.
"""

import logging
import sys
import traceback
from matchms.cli.output import OutputContext


logger = logging.getLogger("matchms.cli")


class CliError(Exception):
    """A CLI failure that can be reported as structured JSON."""

    def __init__(
        self,
        message: str,
        code: str,
        *,
        operation: str = "",
        input_file: str | None = None,
        parameter: str | None = None,
        valid_values: list | None = None,
        output_produced: list | None = None,
        hint: str | None = None,
        exit_code: int = 1,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.operation = operation
        self.input_file = input_file
        self.parameter = parameter
        self.valid_values = valid_values
        self.output_produced = output_produced
        self.hint = hint
        self.exit_code = exit_code
        self.exc_type = ""

    def to_dict(self) -> dict:
        error = {
            "ok": False,
            "error": self.code,
            "message": self.message,
        }
        if self.operation:
            error["operation"] = self.operation
        if self.input_file is not None:
            error["input_file"] = self.input_file
        if self.parameter is not None:
            error["parameter"] = self.parameter
        if self.valid_values is not None:
            error["valid_values"] = self.valid_values
        if self.hint is not None:
            error["hint"] = self.hint
        error["output_produced"] = {
            "any": bool(self.output_produced),
            "files": list(self.output_produced or []),
        }
        if self.exit_code != 1:
            error["exit_code"] = self.exit_code
        return error


def raise_for_unknown_value(
    operation: str,
    parameter: str,
    value: str,
    valid: list[str] | dict,
    kind: str,
    hint: str | None = None,
) -> None:
    """Raise a CliError listing the valid options for an unknown name."""
    valid_values = sorted(valid.keys()) if isinstance(valid, dict) else sorted(valid)
    suggestion = _closest(value, valid_values)
    if suggestion is not None and hint is None:
        hint = f"Did you mean '{suggestion}'?"
    raise CliError(
        f"Unknown {kind} '{value}'.",
        code="unknown_value",
        operation=operation,
        parameter=parameter,
        valid_values=valid_values,
        hint=hint,
    )


def _closest(value: str, candidates: list[str]) -> str | None:
    """Case-insensitive nearest neighbour by Levenshtein distance (<= 2)."""
    best, best_dist = None, 3
    value_l = value.lower()
    for candidate in candidates:
        for alt in (candidate, candidate.lower()):
            dist = _levenshtein(value_l, alt)
            if dist < best_dist:
                best, best_dist = candidate, dist
    return best


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def configure_logging(level: int) -> None:
    """Route the matchms logger (and the tqdm progress bars) to stderr.

    Any handler previously attached to the matchms logger (including the
    stdout handler installed by :func:`matchms.logging_functions._init_logger`)
    is replaced so that CLI output on stdout stays machine-readable.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    cli_logger = logging.getLogger("matchms.cli")
    cli_logger.handlers[:] = []
    cli_logger.setLevel(level)
    cli_logger.propagate = False
    cli_logger.addHandler(handler)
    matchms_logger = logging.getLogger("matchms")
    matchms_logger.handlers[:] = []
    matchms_logger.setLevel(level)
    matchms_logger.addHandler(handler)
    logging.basicConfig(level=level, stream=sys.stderr, force=False)
    # tqdm already writes to stderr; nothing to reconfigure.


def format_traceback(exc: BaseException) -> str:
    """Formats a traceback to display as CLIError"""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def unexpected_error_payload(exc: BaseException, operation: str) -> dict:
    """Build the JSON payload for an error that is not a known CliError."""
    return {
        "ok": False,
        "error": "internal_error",
        "message": f"Unexpected {type(exc).__name__}: {exc}",
        "operation": operation,
        "hint": "Report this issue with the traceback shown here.",
        "output_produced": {"any": False, "files": []},
        "traceback": format_traceback(exc),
    }


def emit_error(ctx: OutputContext, payload: dict) -> None:
    """Emit a structured error: JSON on stdout, short line on stderr."""

    ctx.write_json(payload)
    short = f"ERROR [{payload.get('error')}]: {payload.get('message')}"
    if payload.get("hint"):
        short += f" Hint: {payload['hint']}"
    sys.stderr.write(short + "\n")
