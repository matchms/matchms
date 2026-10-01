"""Typed parsing of ``--param name=value`` pairs.

Values are inferred from their string form so that agent-supplied parameters
are converted correctly (e.g. ``tol=0.1`` becomes a float, ``use_hungarian=true``
a bool, ``top_k=100`` an int).
"""

import inspect
import json
from matchms.cli.errors import CliError
import numpy as np


def parse_params(pairs: list[str] | None, parameter: str = "--param", operation: str = "") -> dict:
    """Parse a list of ``key=value`` strings into a typed dict.

    Raises ``CliError`` (code ``invalid_parameter``) on malformed entries so
    the agent learns the expected syntax.
    """

    params: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise CliError(
                f"Malformed {parameter} entry '{pair}'. Expected syntax: --param name=value",
                code="invalid_parameter",
                operation=operation,
                parameter=parameter,
                valid_values=["key=value"],
                hint="Examples: --param tolerance=0.1 --param use_hungarian=true",
            )
        key, _, raw = pair.partition("=")
        key = key.strip()
        if not key:
            raise CliError(
                f"Malformed {parameter} entry '{pair}': empty parameter name.",
                code="invalid_parameter",
                operation=operation,
                parameter=parameter,
            )
        params[key] = convert_value(key, raw.strip())
    return params


def convert_value(key: str, raw: str):
    """Infer a Python type for a raw string value.

    Falls back to the original string when nothing else matches (so unknown
    parameters fail later with the target's own, more specific error message).
    """
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("none", "null"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith(("[", "{")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def describe_signature(callable_obj) -> dict:
    """Build a JSON-safe description of a callable's signature."""

    params = {}
    required = []
    for name, p in inspect.signature(callable_obj).parameters.items():
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            params[name] = {"type": "variadic positional", "required": False, "default": None}
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            params[name] = {"type": "keyword variadic", "required": False, "default": None}
        else:
            has_default = p.default is not inspect.Parameter.empty
            info = {"type": _type_name(p.annotation), "required": not has_default}
            if has_default:
                info["default"] = _safe_default(p.default)
            params[name] = info
            if not has_default:
                required.append(name)
    return {"parameters": params, "required": required}


def _type_name(annotation) -> str:
    if annotation is None or annotation is inspect.Parameter.empty:
        return "any"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _safe_default(value) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray, np.dtype)):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
