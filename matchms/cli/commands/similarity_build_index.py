"""CLI command: matchms similarity build-index.

Builds a reusable library index from a spectra file and saves it to disk so a
later ``matchms similarity search`` does not have to re-prepare the reference
library. Stdout carries only a summary (JSON or human-readable).

The index is bound to the similarity configuration it was built with: a later
search must use the same ``--method`` and ``--param`` values (checked when the
index is loaded). The index contains only the reference spectra (no queries)
and preserves spectrum order: reference positions in search results refer to
the row order of the original library file, so keep the library file alongside
the index when readable identifiers are needed.
"""

import inspect
import os
import time
from matchms.cli.errors import CliError, raise_for_unknown_value
from matchms.cli.params import describe_signature, parse_params
from matchms.importing import load_ms2_dataset
from matchms.importing.load_spectra import SUPPORTED_FILE_FORMATS as INPUT_FORMATS
from matchms.similarity import __all__ as SIMILARITY_NAMES
from matchms.similarity import get_similarity_function_by_name


CLI_COMMAND = "similarity build-index"

OUTPUT_SUFFIX = ".index.npz"

# Similarity classes that can build a reusable library index (they implement
# build_index() and save_index(); the rest only compute pair/matrix scores).
INDEX_CAPABLE_NAMES = ("Cosine", "Entropy", "EntropySearch", "ModifiedCosine")

# Entropy matching modes accepted by build_index (validated before the
# constructor so the error code is invalid_parameter, not a constructor error).
ENTROPY_MATCHING_MODES = ("fragment", "neutral_loss", "hybrid")

# EntropySearch peak-separation modes that are index-capable ("raise" is
# reported as invalid_input when a library violates the separation requirement).
ENTROPY_SEARCH_PEAK_SEPARATIONS = ("merge", "raise")


def _extension(path: str) -> str | None:
    return os.path.splitext(path)[1].lower().lstrip(".") or None


def _validate_output(args, operation: str) -> None:
    """The output must end in .index.npz and its directory must exist and be writable."""
    output = args.output
    if not output.lower().endswith(OUTPUT_SUFFIX):
        raise CliError(
            f"Output file '{output}' must end in '{OUTPUT_SUFFIX}'.",
            code="invalid_parameter",
            operation=operation,
            parameter="output",
            valid_values=[OUTPUT_SUFFIX],
            hint="Indexes are stored as FlashIndex NPZ archives, e.g. library.index.npz.",
        )
    out_dir = os.path.dirname(os.path.abspath(output))
    if not os.path.isdir(out_dir):
        raise CliError(
            f"The output directory '{out_dir}' of '{output}' does not exist.",
            code="save_failed",
            operation=operation,
            input_file=output,
            hint="Create the directory first; the output directory must exist.",
        )
    if not os.access(out_dir, os.R_OK | os.W_OK):
        raise CliError(
            f"The output directory '{out_dir}' of '{output}' is not writable.",
            code="save_failed",
            operation=operation,
            input_file=output,
            hint="Choose an output directory that exists and is writable.",
        )


def _validate_input(path: str, operation: str) -> tuple[str, str]:
    """Check the library file exists with a supported extension; return (path, format)."""
    if not os.path.exists(path):
        raise CliError(
            f"The specified input file: {path} does not exist.",
            code="input_not_found",
            operation=operation,
            input_file=path,
            valid_values=sorted(INPUT_FORMATS),
            hint="Expected a reference spectra file with a supported extension "
            "(e.g. .mgf, .msp, .mzml, .mzxml, .json, .pickle).",
        )
    file_format = _extension(path)
    if file_format is None or file_format not in INPUT_FORMATS:
        raise CliError(
            f"Input file extension '.{file_format}' of {path} is not a supported input format. "
            "The input format is detected from the file extension only, so files with a "
            "non-standard extension cannot be loaded.",
            code="unsupported_format",
            operation=operation,
            input_file=path,
            valid_values=sorted(INPUT_FORMATS),
            hint="Use a supported extension such as .mgf, .msp, .mzml, .mzxml, .json or .pickle.",
        )
    return path, file_format


def _resolve_method(name: str, operation: str) -> tuple[str, type]:
    """Resolve --method to (canonical name, class); case-insensitive.

    An unknown name raises ``unknown_value`` with the index-capable names and a
    "Did you mean" suggestion.
    """
    lowered = name.lower()
    for candidate in SIMILARITY_NAMES:
        if candidate.lower() == lowered:
            return candidate, get_similarity_function_by_name(candidate)
    raise_for_unknown_value(
        operation=operation,
        parameter="method",
        value=name,
        valid=sorted(INDEX_CAPABLE_NAMES),
        kind="index-capable similarity method",
    )


def _check_index_capable(cls, name: str, operation: str) -> None:
    """A known similarity without index support is rejected (e.g. CosineGreedy)."""
    if cls.__name__ not in INDEX_CAPABLE_NAMES:
        raise CliError(
            f"The similarity '{name}' does not support library indices (it does not implement build_index()).",
            code="unsupported_method",
            operation=operation,
            parameter="method",
            valid_values=sorted(INDEX_CAPABLE_NAMES),
            hint="Index-capable methods: "
            + ", ".join(INDEX_CAPABLE_NAMES)
            + ". See `matchms similarity list` for all methods.",
        )


def _build_params(args, operation: str) -> dict:
    """Merge --param NAME=VALUE pairs and the --tolerance shorthand."""
    params = dict(parse_params(args.param, operation=operation))
    if args.tolerance is not None:
        if "tolerance" in params:
            raise CliError(
                "--tolerance and --param tolerance=... may not be combined.",
                code="invalid_parameter",
                operation=operation,
                parameter="tolerance",
                hint="Use either the --tolerance shorthand or --param tolerance=..., not both.",
            )
        params["tolerance"] = args.tolerance
    return params


def _check_params_accepted(cls, name: str, params: dict, operation: str) -> None:
    """Raise ``invalid_parameter`` when a parameter name is not in the constructor."""
    signature = inspect.signature(cls)
    var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    known = list(signature.parameters)
    for key in params:
        if var_keyword or key in known:
            continue
        raise CliError(
            f"The similarity '{name}' does not accept the parameter '{key}'.",
            code="invalid_parameter",
            operation=operation,
            parameter=key,
            valid_values=known,
            hint=f"See `matchms similarity info {name}` for the available parameters.",
        )


def _effective_params(cls, params: dict) -> dict:
    """Constructor parameters that will actually be used (defaults + overrides)."""
    effective = {
        pname: spec["default"] for pname, spec in describe_signature(cls)["parameters"].items() if "default" in spec
    }
    effective.update(params)
    return effective


def _check_method_params(name: str, params: dict, operation: str) -> None:
    """Method-specific, parameter-level checks (run before any file is loaded).

    - Cosine / ModifiedCosine with use_hungarian=True are not index-capable.
    - EntropySearch peak_separation must be merge or raise.
    - Entropy matching_mode must be fragment, neutral_loss or hybrid.
    """
    if name in ("Cosine", "ModifiedCosine") and params.get("use_hungarian"):
        raise CliError(
            f"The similarity '{name}' does not support library indices with "
            "use_hungarian=True (optimal-assignment scoring has no persistent index).",
            code="unsupported_method",
            operation=operation,
            parameter="use_hungarian",
            valid_values=["use_hungarian=false"],
            hint="Drop the parameter (default use_hungarian=false) or use `similarity matrix`.",
        )
    if name == "EntropySearch":
        if params.get("use_ppm"):
            raise CliError(
                "EntropySearch only supports an absolute (Da) tolerance, not ppm.",
                code="invalid_parameter",
                operation=operation,
                parameter="use_ppm",
                valid_values=["use_ppm=false"],
                hint="Pass --param use_ppm=false, or use `--method Entropy` for ppm matching.",
            )
        if params.get("peak_separation", "merge") not in ENTROPY_SEARCH_PEAK_SEPARATIONS:
            raise CliError(
                "EntropySearch peak_separation must be 'merge' or 'raise'.",
                code="invalid_parameter",
                operation=operation,
                parameter="peak_separation",
                valid_values=list(ENTROPY_SEARCH_PEAK_SEPARATIONS),
                hint="Pass --param peak_separation=merge (default) or --param peak_separation=raise.",
            )
    if name == "Entropy" and params.get("matching_mode", "fragment") not in ENTROPY_MATCHING_MODES:
        raise CliError(
            "Entropy matching_mode must be 'fragment', 'neutral_loss' or 'hybrid'.",
            code="invalid_parameter",
            operation=operation,
            parameter="matching_mode",
            valid_values=list(ENTROPY_MATCHING_MODES),
            hint="Pass --param matching_mode=fragment (default) for fragment-only matching.",
        )


def _instantiate(cls, name: str, params: dict, operation: str):
    """Build the similarity instance, translating validation errors to CliError."""
    try:
        return cls(**params)
    except (TypeError, ValueError) as exc:
        raise CliError(
            f"Invalid parameter values for the similarity '{name}': {exc}",
            code="invalid_parameter",
            operation=operation,
            parameter=name,
            hint=f"See `matchms similarity info {name}` for the parameters and their expected values.",
        ) from exc


def _load_collection(path: str, file_format: str, operation: str):
    try:
        return load_ms2_dataset(path, ftype=file_format)
    except ValueError as exc:
        if "at least one Spectrum" in str(exc):
            raise CliError(
                f"No spectra were loaded from {path}; nothing to index.",
                code="empty_spectra",
                operation=operation,
                input_file=path,
            ) from exc
        raise CliError(
            f"Failed to load spectra from {path}: {exc}",
            code="compute_error",
            operation=operation,
            input_file=path,
        ) from exc
    except Exception as exc:
        raise CliError(
            f"Failed to load spectra from {path}: {exc}",
            code="compute_error",
            operation=operation,
            input_file=path,
        ) from exc


def _build_index(similarity, name: str, collection, operation: str):
    """Build the index, mapping the known failure modes to structured errors."""
    try:
        return similarity.build_index(collection)
    except (ModuleNotFoundError, ImportError) as exc:
        raise CliError(
            f"A required dependency of the similarity '{name}' is not installed: {exc}",
            code="missing_dependency",
            operation=operation,
            parameter=name,
            hint="Install the missing dependency and retry.",
        ) from exc
    except ValueError as exc:
        message = str(exc)
        if "use peak_separation='merge'" in message or "peak_separation='merge'" in message:
            raise CliError(
                f"The library violates the peak-separation requirement of '{name}' (peak_separation=raise): {message}",
                code="invalid_input",
                operation=operation,
                hint="Rebuild with --param peak_separation=merge (close peaks are merged) "
                "or preprocess the library so no two peaks are closer than 2 * max_tolerance.",
            ) from exc
        raise CliError(
            f"Failed to build the index with the similarity '{name}': {exc}",
            code="compute_error",
            operation=operation,
            hint="Check the similarity method and its parameters, and the input spectra.",
        ) from exc
    except Exception as exc:
        raise CliError(
            f"Failed to build the index with the similarity '{name}': {exc}",
            code="compute_error",
            operation=operation,
            hint="Check the similarity method and its parameters, and the input spectra.",
        ) from exc


def _save_index(similarity, name: str, index, output: str, operation: str) -> None:
    try:
        similarity.save_index(index, output, overwrite=True)
    except (ModuleNotFoundError, ImportError) as exc:
        raise CliError(
            f"A required dependency of the similarity '{name}' is not installed: {exc}",
            code="missing_dependency",
            operation=operation,
            input_file=output,
            hint="Install the missing dependency and retry.",
        ) from exc
    except Exception as exc:
        raise CliError(
            f"Failed to save the index to {output}: {exc}",
            code="save_failed",
            operation=operation,
            input_file=output,
        ) from exc


def run(args, ctx) -> int:
    """Run the `similarity build-index` command."""
    operation = CLI_COMMAND
    _validate_output(args, operation)
    library, file_format = _validate_input(args.library, operation)

    method_name, cls = _resolve_method(args.method, operation)
    _check_index_capable(cls, method_name, operation)
    params = _build_params(args, operation)
    _check_params_accepted(cls, method_name, params, operation)
    _check_method_params(method_name, _effective_params(cls, params), operation)

    collection = _load_collection(library, file_format, operation)
    n_spectra = len(collection)

    similarity = _instantiate(cls, method_name, params, operation)

    started = time.perf_counter()
    index = _build_index(similarity, method_name, collection, operation)
    elapsed = time.perf_counter() - started

    _save_index(similarity, method_name, index, args.output, operation)

    payload = {
        "ok": True,
        "operation": CLI_COMMAND,
        "method": {
            "name": method_name,
            "class": cls.__name__,
            "params": _effective_params(cls, params),
        },
        "library": {"file": library, "n_spectra": n_spectra},
        "index": {
            "file": args.output,
            "format": "index.npz",
            "size_bytes": os.path.getsize(args.output),
        },
        "elapsed_seconds": round(elapsed, 6),
        "note": (
            "Use the same --method and --param values with 'similarity search'. "
            "The index contains only the reference spectra (no queries) and preserves "
            "spectrum order; keep the library file alongside the index for readable identifiers."
        ),
    }

    if ctx.machine_mode:
        ctx.write_json(payload)
    else:
        ctx.write_text(_format_human(payload))
    return 0


def _format_human(payload: dict) -> str:
    method = payload["method"]
    lines = [
        f"Library index: {payload['library']['file']}",
        f"  method:   {method['name']} ({', '.join(f'{k}={v}' for k, v in method['params'].items())})",
        f"  library:  {payload['library']['n_spectra']} spectra",
        f"  index:    {payload['index']['file']} "
        f"({payload['index']['format']}, {payload['index']['size_bytes']} bytes)",
        f"  elapsed:  {payload['elapsed_seconds']:.3f} s",
        "",
        payload["note"],
        "",
        "Run with --json for the machine-readable version.",
    ]
    return "\n".join(lines)
