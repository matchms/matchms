"""Output conventions for the matchms CLI.

Rules
-----
- Data commands (list, describe, show) default to structured JSON on stdout
  when stdout is not an interactive terminal (agent mode). Pass ``--json`` to
  force JSON and ``--table`` to force human-readable output.
- Artifacts are always written to files; stdout only summarizes them.
- Progress bars and log messages go to stderr only.
"""

import json
import sys
from typing import Any, TextIO
import numpy as np


# Schema version of the CLI's JSON output. Bump when the structure of any
# JSON payload changes in a way that consumers must be aware of.
CLI_JSON_SCHEMA_VERSION = "1.0"


class OutputContext:
    """Resolves output mode and carries per-invocation options."""

    def __init__(
        self,
        json_flag: bool | None,
        table_flag: bool | None,
        quiet: bool,
        verbose: bool,
        stream: TextIO | None = None,
    ):
        self.json_flag = json_flag
        self.table_flag = table_flag
        self.quiet = quiet
        self.verbose = verbose
        self.stream: TextIO = stream if stream is not None else sys.stdout

    @property
    def machine_mode(self) -> bool:
        """True when the machine (JSON) output format is selected.

        Precedence: explicit ``--json`` > explicit ``--table`` > isatty
        auto-detection (JSON when stdout is a pipe/file).
        """
        if self.json_flag:
            return True
        if self.table_flag:
            return False
        return not self.stream.isatty()

    def write_json(self, payload: dict[str, Any]) -> None:
        self.stream.write(json.dumps(payload, indent=2, default=_json_default))
        self.stream.write("\n")

    def write_text(self, text: str) -> None:
        if text:
            self.stream.write(text if text.endswith("\n") else text + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    return str(value)


def format_float(value: Any, precision: int = 6) -> str:
    """Formats a given value as a float with a specified precision."""
    try:
        return f"{float(value):.{precision}g}"
    except (TypeError, ValueError):
        return str(value)


def human_table(columns: list[str], rows: list[list[Any]]) -> str:
    """Render a simple fixed-width table without third-party dependencies."""
    str_rows = [[_cell_str(cell) for cell in row] for row in rows]
    widths = [len(col) for col in columns]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = ["  ".join(col.ljust(widths[i]) for i, col in enumerate(columns))]
    lines.append("  ".join("-" * w for w in widths))
    for row in str_rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def indent_text(text: str, prefix: str = "  ") -> str:
    """Indent every line of *text* with *prefix* for the human-readable output."""
    return "\n".join(prefix + line for line in text.splitlines())


def parameters_table(parameters: list[dict]) -> str:
    """Render a list of parameter entries (see :func:`introspection.signature_parameters`).

    Used by the ``info`` commands to show name, type, required flag, default
    and description of one object's constructor / call signature.
    """
    return human_table(
        ["parameter", "type", "required", "default", "description"],
        [[p["name"], p["type"], p["required"], p.get("default"), p.get("description", "")] for p in parameters],
    )


def _cell_str(cell: Any) -> str:
    if cell is None:
        return ""
    if isinstance(cell, bool):
        return "true" if cell else "false"
    if isinstance(cell, float):
        return format_float(cell)
    if isinstance(cell, (list, tuple)):
        return ", ".join(_cell_str(c) for c in cell)
    if isinstance(cell, dict):
        return "; ".join(f"{k}={_cell_str(v)}" for k, v in cell.items())
    return str(cell)
