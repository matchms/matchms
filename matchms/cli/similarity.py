"""Shared similarity helpers for the matchms CLI.

The three similarity commands (``matrix``, ``build-index``, ``search``) all need
to resolve a ``--method`` name to a similarity class, merge ``--param``/
``--tolerance`` into constructor arguments, validate that the constructor
accepts them, instantiate the class, and translate constructor failures into
structured :class:`~matchms.cli.errors.CliError` errors.

Those steps are identical across the commands, so they live here once. Where
the commands genuinely differ, the difference is a *parameter*, not a fork of
the code:

- :func:`resolve_method` takes the set of accepted names (all similarities for
  ``matrix``; the index-capable ones for ``build-index``/``search``) and a label
  for the "Did you mean" suggestion;
- :func:`instantiate` takes ``fingerprint_hint`` so only ``matrix`` (which can
  run a dense FingerprintSimilarity) gets the actionable RDKit message;
- the method-specific parameter rules differ between ``matrix`` and the
  index-capable commands, so :func:`check_index_method_params` holds the shared
  rules for ``build-index``/``search`` and ``matrix`` keeps its own.
"""

import inspect
from matchms.cli.constants import (
    ERROR_INVALID_PARAMETER,
    ERROR_UNSUPPORTED_METHOD,
)
from matchms.cli.errors import CliError, raise_for_unknown_value
from matchms.cli.introspection import similarity_methods
from matchms.cli.params import parse_params
from matchms.similarity import __all__ as SIMILARITY_NAMES
from matchms.similarity import get_similarity_function_by_name


def _index_capable_names() -> tuple[str, ...]:
    """Similarity classes that can build a reusable library index.

    Derived from the same :func:`similarity_methods` introspection that
    ``similarity list`` uses, so the index/search commands and ``similarity list``
    can never disagree: a class is index-capable exactly when it implements
    ``build_index()`` (which in matchms always comes with the indexed search
    workflow). The result is sorted so the valid-value lists are deterministic.
    """
    names = [
        name for name in SIMILARITY_NAMES if "build_index" in similarity_methods(get_similarity_function_by_name(name))
    ]
    return tuple(sorted(names))


# Similarity classes that can build a reusable library index (they implement
# build_index() and save_index(); the rest only compute pair/matrix scores).
INDEX_CAPABLE_NAMES = _index_capable_names()

# Entropy matching modes accepted by an index (validated before the constructor
# so the error code is invalid_parameter, not a constructor error).
ENTROPY_MATCHING_MODES = ("fragment", "neutral_loss", "hybrid")

# EntropySearch peak-separation modes that are index-capable ("raise" is
# reported as invalid_input when a library violates the separation requirement).
ENTROPY_SEARCH_PEAK_SEPARATIONS = ("merge", "raise")


def resolve_method(name: str, operation: str, *, valid: list, kind: str) -> tuple[str, type]:
    """Resolve a ``--method`` value to ``(canonical name, class)``.

    Case-insensitive. An unknown name raises ``unknown_value`` with *valid*
    (the accepted names for this command) and a "Did you mean" suggestion.
    """
    lowered = name.lower()
    for candidate in SIMILARITY_NAMES:
        if candidate.lower() == lowered:
            return candidate, get_similarity_function_by_name(candidate)
    raise_for_unknown_value(
        operation=operation,
        parameter="method",
        value=name,
        valid=valid,
        kind=kind,
    )


def build_params(args, operation: str) -> dict:
    """Merge ``--param NAME=VALUE`` pairs and the ``--tolerance`` shorthand."""
    params = dict(parse_params(args.param, operation=operation))
    if args.tolerance is not None:
        if "tolerance" in params:
            raise CliError(
                "--tolerance and --param tolerance=... may not be combined.",
                code=ERROR_INVALID_PARAMETER,
                operation=operation,
                parameter="tolerance",
                hint="Use either the --tolerance shorthand or --param tolerance=..., not both.",
            )
        params["tolerance"] = args.tolerance
    return params


def check_params_accepted(cls, name: str, params: dict, operation: str) -> None:
    """Raise ``invalid_parameter`` when a parameter name is not in the constructor."""
    signature = inspect.signature(cls)
    var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    known = list(signature.parameters)
    for key in params:
        if var_keyword or key in known:
            continue
        raise CliError(
            f"The similarity '{name}' does not accept the parameter '{key}'.",
            code=ERROR_INVALID_PARAMETER,
            operation=operation,
            parameter=key,
            valid_values=known,
            hint=f"See `matchms similarity info {name}` for the available parameters.",
        )


def effective_params(sig: dict, params: dict) -> dict:
    """Constructor parameters that will actually be used (defaults + overrides).

    *sig* is a :func:`~matchms.cli.params.describe_signature` result; compute it
    once per run and share it with :func:`check_missing_required`.
    """
    effective = {pname: spec["default"] for pname, spec in sig["parameters"].items() if "default" in spec}
    effective.update(params)
    return effective


def check_missing_required(name: str, sig: dict, params: dict, operation: str, *, skip: tuple = ()) -> None:
    """Raise ``invalid_parameter`` when a required constructor parameter is missing.

    Parameters in *skip* are not expected on the command line (e.g.
    FingerprintSimilarity's RDKit ``fingerprint_generator``).
    """
    required = [p for p in sig["required"] if p not in skip]
    missing = [p for p in required if p not in params]
    if missing:
        raise CliError(
            f"The similarity '{name}' is missing required parameter(s): {', '.join(missing)}.",
            code=ERROR_INVALID_PARAMETER,
            operation=operation,
            parameter=missing[0],
            valid_values=required,
            hint=f"Pass them with --param {missing[0]}=value. "
            f"See `matchms similarity info {name}` for the required parameters.",
        )


def check_index_capable(cls, name: str, operation: str) -> None:
    """A known similarity without index support is rejected (e.g. CosineGreedy)."""
    if cls.__name__ not in INDEX_CAPABLE_NAMES:
        raise CliError(
            f"The similarity '{name}' does not support library indices (it does not implement build_index()).",
            code=ERROR_UNSUPPORTED_METHOD,
            operation=operation,
            parameter="method",
            valid_values=sorted(INDEX_CAPABLE_NAMES),
            hint="Index-capable methods: "
            + ", ".join(INDEX_CAPABLE_NAMES)
            + ". See `matchms similarity list` for all methods.",
        )


def check_index_method_params(name: str, params: dict, operation: str) -> None:
    """Method-specific, parameter-level checks for the index-capable commands.

    Shared by ``build-index`` and ``search`` (both run an index):

    - Cosine / ModifiedCosine with ``use_hungarian=True`` are not index-capable.
    - EntropySearch ``peak_separation`` must be ``merge`` or ``raise``.
    - Entropy ``matching_mode`` must be ``fragment``, ``neutral_loss`` or
      ``hybrid``.
    """
    if name in ("Cosine", "ModifiedCosine") and params.get("use_hungarian"):
        raise CliError(
            f"The similarity '{name}' does not support library indices with "
            "use_hungarian=True (optimal-assignment scoring has no persistent index).",
            code=ERROR_UNSUPPORTED_METHOD,
            operation=operation,
            parameter="use_hungarian",
            valid_values=["use_hungarian=false"],
            hint="Drop the parameter (default use_hungarian=false) or use `similarity matrix`.",
        )
    if name == "EntropySearch":
        if params.get("use_ppm"):
            raise CliError(
                "EntropySearch only supports an absolute (Da) tolerance, not ppm.",
                code=ERROR_INVALID_PARAMETER,
                operation=operation,
                parameter="use_ppm",
                valid_values=["use_ppm=false"],
                hint="Pass --param use_ppm=false, or use `--method Entropy` for ppm matching.",
            )
        if params.get("peak_separation", "merge") not in ENTROPY_SEARCH_PEAK_SEPARATIONS:
            raise CliError(
                "EntropySearch peak_separation must be 'merge' or 'raise'.",
                code=ERROR_INVALID_PARAMETER,
                operation=operation,
                parameter="peak_separation",
                valid_values=list(ENTROPY_SEARCH_PEAK_SEPARATIONS),
                hint="Pass --param peak_separation=merge (default) or --param peak_separation=raise.",
            )
    if name == "Entropy" and params.get("matching_mode", "fragment") not in ENTROPY_MATCHING_MODES:
        raise CliError(
            "Entropy matching_mode must be 'fragment', 'neutral_loss' or 'hybrid'.",
            code=ERROR_INVALID_PARAMETER,
            operation=operation,
            parameter="matching_mode",
            valid_values=list(ENTROPY_MATCHING_MODES),
            hint="Pass --param matching_mode=fragment (default) for fragment-only matching.",
        )


def instantiate(cls, name: str, params: dict, operation: str, *, fingerprint_hint: bool = False):
    """Build the similarity instance, translating constructor errors to CliError.

    With ``fingerprint_hint`` set (only ``matrix`` runs a dense
    FingerprintSimilarity), a missing RDKit ``fingerprint_generator`` gets a
    dedicated, actionable message; otherwise constructor failures are reported
    generically.
    """
    try:
        return cls(**params)
    except TypeError as exc:
        if fingerprint_hint and cls.__name__ == "FingerprintSimilarity":
            raise CliError(
                "FingerprintSimilarity requires a fingerprint_generator (an RDKit fingerprint "
                "generator object) that cannot be passed via --param, so it cannot be run "
                "directly from the CLI.",
                code=ERROR_INVALID_PARAMETER,
                operation=operation,
                parameter="fingerprint_generator",
                hint="Compute fingerprint similarity programmatically with "
                "matchms.similarity.FingerprintSimilarity, or use a peak/metadata-based method.",
            ) from exc
        raise CliError(
            f"Invalid parameter values for the similarity '{name}': {exc}",
            code=ERROR_INVALID_PARAMETER,
            operation=operation,
            parameter=name,
            hint=f"See `matchms similarity info {name}` for the parameters and their expected values.",
        ) from exc
    except ValueError as exc:
        raise CliError(
            f"The similarity '{name}' rejected its parameters: {exc}",
            code=ERROR_INVALID_PARAMETER,
            operation=operation,
            parameter=name,
            hint=f"See `matchms similarity info {name}` for the parameters and their expected values.",
        ) from exc


__all__ = [
    "ENTROPY_MATCHING_MODES",
    "ENTROPY_SEARCH_PEAK_SEPARATIONS",
    "INDEX_CAPABLE_NAMES",
    "build_params",
    "check_index_capable",
    "check_index_method_params",
    "check_missing_required",
    "check_params_accepted",
    "effective_params",
    "instantiate",
    "resolve_method",
]
