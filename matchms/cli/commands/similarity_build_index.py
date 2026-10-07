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

import os
import time
from matchms.cli import files, similarity
from matchms.cli.constants import (
    ERROR_COMPUTE_ERROR,
    ERROR_INVALID_INPUT,
    ERROR_INVALID_PARAMETER,
    ERROR_MISSING_DEPENDENCY,
    ERROR_SAVE_FAILED,
)
from matchms.cli.errors import CliError
from matchms.cli.params import describe_signature


CLI_COMMAND = "similarity build-index"

OUTPUT_SUFFIX = ".index.npz"

INDEX_CAPABLE_NAMES = similarity.INDEX_CAPABLE_NAMES


def _validate_input(path: str, operation: str) -> tuple[str, str]:
    """Check the library file exists with a supported extension; return (path, format)."""
    file_format = files.validate_input_file(
        path,
        operation,
        kind="library file",
        hint="Expected a reference spectra file with a supported extension "
        "(e.g. .mgf, .msp, .mzml, .mzxml, .json, .pickle).",
    )
    return path, file_format


def _resolve_method(name: str, operation: str) -> tuple[str, type]:
    """Resolve --method to (canonical name, class); only index-capable names."""
    return similarity.resolve_method(
        name,
        operation,
        valid=sorted(INDEX_CAPABLE_NAMES),
        kind="index-capable similarity method",
    )


def _validate_output(args, operation: str) -> None:
    """The output must end in .index.npz and its directory must exist and be writable."""
    output = args.output
    if not output.lower().endswith(OUTPUT_SUFFIX):
        raise CliError(
            f"Output file '{output}' must end in '{OUTPUT_SUFFIX}'.",
            code=ERROR_INVALID_PARAMETER,
            operation=operation,
            parameter="output",
            valid_values=[OUTPUT_SUFFIX],
            hint="Indexes are stored as FlashIndex NPZ archives, e.g. library.index.npz.",
        )
    files.validate_output_dir(output, operation)


def _build_index(similarity_instance, name: str, collection, operation: str):
    """Build the index, mapping the known failure modes to structured errors."""
    try:
        return similarity_instance.build_index(collection)
    except (ModuleNotFoundError, ImportError) as exc:
        raise CliError(
            f"A required dependency of the similarity '{name}' is not installed: {exc}",
            code=ERROR_MISSING_DEPENDENCY,
            operation=operation,
            parameter=name,
            hint="Install the missing dependency and retry.",
        ) from exc
    except ValueError as exc:
        message = str(exc)
        if "peak_separation='merge'" in message:
            raise CliError(
                f"The library violates the peak-separation requirement of '{name}' (peak_separation=raise): {message}",
                code=ERROR_INVALID_INPUT,
                operation=operation,
                hint="Rebuild with --param peak_separation=merge (close peaks are merged) "
                "or preprocess the library so no two peaks are closer than 2 * max_tolerance.",
            ) from exc
        raise CliError(
            f"Failed to build the index with the similarity '{name}': {exc}",
            code=ERROR_COMPUTE_ERROR,
            operation=operation,
            hint="Check the similarity method and its parameters, and the input spectra.",
        ) from exc
    except Exception as exc:
        raise CliError(
            f"Failed to build the index with the similarity '{name}': {exc}",
            code=ERROR_COMPUTE_ERROR,
            operation=operation,
            hint="Check the similarity method and its parameters, and the input spectra.",
        ) from exc


def _save_index(similarity_instance, name: str, index, output: str, operation: str) -> None:
    try:
        similarity_instance.save_index(index, output, overwrite=True)
    except (ModuleNotFoundError, ImportError) as exc:
        raise CliError(
            f"A required dependency of the similarity '{name}' is not installed: {exc}",
            code=ERROR_MISSING_DEPENDENCY,
            operation=operation,
            input_file=output,
            hint="Install the missing dependency and retry.",
        ) from exc
    except Exception as exc:
        raise CliError(
            f"Failed to save the index to {output}: {exc}",
            code=ERROR_SAVE_FAILED,
            operation=operation,
            input_file=output,
        ) from exc


def run(args, ctx) -> int:
    """Run the `similarity build-index` command."""
    operation = CLI_COMMAND
    _validate_output(args, operation)
    library, file_format = _validate_input(args.library, operation)

    method_name, cls = _resolve_method(args.method, operation)
    similarity.check_index_capable(cls, method_name, operation)
    params = similarity.build_params(args, operation)
    signature = describe_signature(cls)
    similarity.check_params_accepted(cls, method_name, params, operation)
    effective = similarity.effective_params(signature, params)
    similarity.check_index_method_params(method_name, effective, operation)

    collection = files.load_collection(library, file_format, operation)
    n_spectra = len(collection)

    similarity_instance = similarity.instantiate(cls, method_name, params, operation)

    started = time.perf_counter()
    index = _build_index(similarity_instance, method_name, collection, operation)
    elapsed = time.perf_counter() - started

    _save_index(similarity_instance, method_name, index, args.output, operation)

    payload = {
        "ok": True,
        "operation": CLI_COMMAND,
        "method": {
            "name": method_name,
            "class": cls.__name__,
            "params": effective,
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
