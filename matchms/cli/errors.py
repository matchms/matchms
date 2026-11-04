"""Structured error handling for the matchms CLI.

The CLI exposes a single, stable machine-readable error envelope. Every
failure — an argparse validation error, a missing command or subcommand, a
bad value or type, a domain error (:class:`CliError`) and any unexpected
exception — is rendered through the same code path into one JSON document on
stdout (in machine mode):

.. code-block:: json

    {
      "schema_version": "1.0.0",
      "ok": false,
      "error": {
        "code": "invalid_value",
        "message": "argument --mode: invalid choice: 'invalid' (choose from 'dense', 'sparse')",
        "operation": "similarity matrix",
        "parameter": "--mode",
        "value": "invalid",
        "valid_values": ["dense", "sparse"],
        "hint": null
      },
      "output_produced": {"any": false, "files": []}
    }

Contract
--------
- ``error.code`` is the stable, semantic API. Consumers branch on it; they
  never parse ``message``. The core vocabulary (lowercase, stable across
  releases) is:

  ======================  ====================================================
  code                    meaning
  ======================  ====================================================
  ``invalid_argument``    unknown flag or unexpected extra argument
  ``missing_argument``    a required argument was not supplied
  ``invalid_value``       an argument value or type is invalid (bad choices,
                          bad int/float conversion, unknown subcommand name)
  ``missing_command``     no top-level command was given
  ``missing_subcommand``  a command group was given without a subcommand
  ``unknown_value``       a *domain* name (method, filter, pipeline, ...) is
                          unknown
  ``file_not_found``      a referenced input/output file does not exist
  ``invalid_input``       input data that is structurally invalid for the
                          operation
  ``permission_denied``   the process lacks access to a file or directory
  ``conflict``            arguments are mutually exclusive / cannot be combined
  ``execution_error``     the operation failed while running (I/O, compute)
  ``internal_error``      an unexpected exception escaped the command
  ======================  ====================================================

  Commands may introduce additional, domain-specific codes (see
  :mod:`matchms.cli.constants`); each failure condition maps to exactly one
  code.
- Exit codes are coarse-grained only: ``0`` success, ``1`` execution or
  internal failure, ``2`` CLI usage / argument validation failure.
- ``output_produced`` always reports the side effects of the run so far, so
  a non-zero exit code does not hide partially written artifacts.
- Raw tracebacks are never exposed by default. When debug mode is explicitly
  enabled (the environment variable ``MATCHMS_CLI_DEBUG`` set to a truthy
  value), the exception type and traceback are attached under the dedicated
  ``debug`` field — never inside ``message``.
"""

import difflib
import logging
import os
import re
import sys
import traceback
from matchms.cli.output import OutputContext


logger = logging.getLogger("matchms.cli")

#: Exit code for unexpected/internal failures.
EXIT_INTERNAL = 1
#: Exit code for CLI usage / argument validation failures.
EXIT_USAGE = 2


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
        value: str | None = None,
        valid_values: list | None = None,
        output_produced: list | None = None,
        hint: str | None = None,
        exit_code: int = EXIT_INTERNAL,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.operation = operation
        self.input_file = input_file
        self.parameter = parameter
        self.value = value
        self.valid_values = valid_values
        self.output_produced = output_produced
        self.hint = hint
        self.exit_code = exit_code

    def to_dict(self) -> dict:
        """Render the stable error envelope (see the module docstring)."""
        return build_error_payload(
            self.code,
            self.message,
            operation=self.operation,
            input_file=self.input_file,
            parameter=self.parameter,
            value=self.value,
            valid_values=self.valid_values,
            output_produced=self.output_produced,
            hint=self.hint,
        )


class ArgparseError(CliError):
    """An argparse validation failure raised instead of ``SystemExit``.

    :class:`~matchms.cli.parser.CliArgumentParser` raises this (with the
    structured details already extracted from the parser's message) so that
    parser errors flow through the same error renderer as runtime errors.
    Always exits with code 2.
    """

    def __init__(self, message: str, code: str, *, operation: str = "", **details):
        super().__init__(
            message,
            code,
            operation=operation,
            exit_code=EXIT_USAGE,
            **details,
        )


def build_error_payload(
    code: str,
    message: str,
    *,
    operation: str = "",
    input_file: str | None = None,
    parameter: str | None = None,
    value: str | None = None,
    valid_values: list | None = None,
    output_produced: list | None = None,
    hint: str | None = None,
    debug: dict | None = None,
) -> dict:
    """Assemble the stable machine-readable error envelope.

    This is the single renderer for every CLI failure, so the envelope shape
    (and the order of the keys) is identical no matter where the error came
    from: the argument parser, a domain :class:`CliError` or an unexpected
    exception.
    """
    error = {
        "code": code,
        "message": message,
    }
    if operation:
        error["operation"] = operation
    if parameter is not None:
        error["parameter"] = parameter
    if value is not None:
        error["value"] = value
    if valid_values is not None:
        error["valid_values"] = list(valid_values)
    error["hint"] = hint
    if input_file is not None:
        error["input_file"] = input_file
    payload = {
        "ok": False,
        "error": error,
        "output_produced": {
            "any": bool(output_produced),
            "files": list(output_produced or []),
        },
    }
    if debug is not None:
        payload["debug"] = debug
    return payload


def debug_payload_enabled() -> bool:
    """True when debug mode is explicitly enabled via ``MATCHMS_CLI_DEBUG``."""
    return os.environ.get("MATCHMS_CLI_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


# --- parser error classification ---------------------------------------------
#
# argparse does not structure its own errors: it only produces a message
# string. The regexes below map those stable, CPython-generated messages onto
# the CLI error vocabulary (see the module docstring).

_RE_UNKNOWN = re.compile(r"unrecognized arguments: (?P<args>.+)")
_RE_REQUIRED = re.compile(r"(?:the following arguments are required|missing required argument): (?P<args>.+)")
_RE_EXPECTED = re.compile(
    r"^(?:argument )?(?P<param>[^:]+): expected (?:one|an integer|an int|an float|a float) argument"
)
_RE_CHOICE = re.compile(
    r"^(?:argument )?(?P<param>[^:]+): invalid choice: (?P<value>.+?)"
    r"(?: \(choose from (?P<choices>.+)\))?$"
)
_RE_TYPE = re.compile(
    r"^(?:argument )?(?P<param>[^:]+): invalid (?P<type>\w+) value: (?P<value>.+?)(?: \(choose from .+\))?$"
)
_RE_PARSER_NAME = re.compile(r"unknown parser (?P<value>['\"])(?P<name>[^'\"]+)\1 \(choices: (?P<choices>.+)\)")


def classify_argparse_error(
    message: str,
    *,
    operation: str = "",
    valid_values: list | None = None,
) -> ArgparseError:
    """Translate an argparse error *message* into a structured :class:`ArgparseError`.

    *valid_values* (when known by the caller, e.g. the choices of a
    subcommand) is exposed as a structured field instead of only being
    embedded in the message.
    """
    code, parameter, value, choices, hint = (
        "invalid_argument",
        None,
        None,
        valid_values,
        None,
    )

    match = _RE_PARSER_NAME.search(message)
    if match:
        code = "invalid_value"
        parameter = "<subcommand>"
        value = match["name"]
        choices = [c.strip() for c in match["choices"].split(", ")]
    elif match := _RE_UNKNOWN.search(message):
        parameter = match["args"].split()[0]
    elif match := _RE_REQUIRED.search(message):
        code = "missing_argument"
        parameter = match["args"].split(",")[0].strip()
        hint = "Re-run the command with the required argument supplied."
    elif match := _RE_EXPECTED.search(message):
        code = "missing_argument"
        parameter = _clean_parameter(match["param"])
        hint = f"The option '{parameter}' requires a value."
    elif match := _RE_CHOICE.search(message):
        code = "invalid_value"
        parameter = _clean_parameter(match["param"])
        value = _clean_value(match["value"])
        if match["choices"]:
            choices = _parse_choices(match["choices"])
    elif match := _RE_TYPE.search(message):
        code = "invalid_value"
        parameter = _clean_parameter(match["param"])
        value = _clean_value(match["value"])

    return ArgparseError(
        message,
        code,
        operation=operation,
        parameter=parameter,
        value=value,
        valid_values=choices,
        hint=hint,
    )


def _clean_parameter(raw: str) -> str | None:
    """Trim argparse boilerplate ('argument X:', 'see usage') off a parameter name."""
    raw = raw.strip().rstrip(":")
    raw = raw.removeprefix("argument ")
    return raw.strip() or None


def _clean_value(raw: str) -> str | None:
    """Strip argparse's single-quoting and trailing colon from a bad value."""
    raw = raw.strip().rstrip(":")
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        raw = raw[1:-1]
    return raw or None


def _parse_choices(raw: str) -> list[str]:
    """Parse argparse's "(choose from ...)" list, handling quoted members."""
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1]
    members = []
    for part in re.split(r",\s*", raw):
        part = part.strip()
        if len(part) >= 2 and part[0] == part[-1] and part[0] in "'\"":
            part = part[1:-1]
        if part:
            members.append(part)
    return members


def raise_for_unknown_value(
    operation: str,
    parameter: str,
    value: str,
    valid: list[str] | dict,
    kind: str,
    hint: str | None = None,
) -> None:
    """Raise a CliError listing the valid options for an unknown domain name."""
    valid_values = sorted(valid.keys()) if isinstance(valid, dict) else sorted(valid)
    suggestion = _closest(value, valid_values)
    if suggestion is not None and hint is None:
        hint = f"Did you mean '{suggestion}'?"
    raise CliError(
        f"Unknown {kind} '{value}'.",
        code="unknown_value",
        operation=operation,
        parameter=parameter,
        value=value,
        valid_values=valid_values,
        hint=hint,
    )


def _closest(value: str, candidates: list[str]) -> str | None:
    """Closest candidate by case-insensitive similarity, or None."""
    if not candidates:
        return None
    matches = difflib.get_close_matches(value.lower(), [candidate.lower() for candidate in candidates], n=1, cutoff=0.6)
    if not matches:
        return None
    for candidate in candidates:
        if candidate.lower() == matches[0]:
            return candidate
    return None


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
    """Build the JSON payload for an error that is not a known CliError.

    By default the traceback is NOT exposed. When debug mode is explicitly
    enabled, the exception type and traceback are attached under the
    dedicated ``debug`` field — never inside ``message``.
    """
    debug = None
    if debug_payload_enabled():
        debug = {
            "exception_type": type(exc).__name__,
            "traceback": format_traceback(exc),
        }
    return build_error_payload(
        "internal_error",
        "An unexpected internal error occurred.",
        operation=operation,
        debug=debug,
    )


def emit_error(ctx: OutputContext, payload: dict) -> None:
    """Emit a structured error: JSON on stdout, concise line on stderr.

    In machine mode the full JSON document goes to stdout and only a concise,
    human-readable summary goes to stderr. In human mode (interactive
    terminal without ``--json``) the human-readable error is kept on stdout
    so terminal users keep the familiar experience; the same concise summary
    is always mirrored to stderr.
    """
    if ctx.machine_mode:
        ctx.write_json(payload)
    else:
        _write_human_error(payload)

    short = f"ERROR [{payload['error']['code']}]: {payload['error']['message']}"
    if payload["error"].get("hint"):
        short += f" Hint: {payload['error']['hint']}"
    sys.stderr.write(short + "\n")


def _write_human_error(payload: dict) -> None:
    error = payload["error"]
    stream = sys.stdout
    stream.write(f"error: {error['message']}\n")
    if error.get("operation"):
        stream.write(f"operation: {error['operation']}\n")
    for key in ("parameter", "value", "input_file"):
        if error.get(key) is not None:
            stream.write(f"{key}: {error[key]}\n")
    if error.get("valid_values"):
        stream.write(f"valid values: {', '.join(str(v) for v in error['valid_values'])}\n")
    if error.get("hint"):
        stream.write(f"hint: {error['hint']}\n")
    produced = payload.get("output_produced") or {}
    if produced.get("any"):
        stream.write(f"output produced: {', '.join(produced['files'])}\n")
