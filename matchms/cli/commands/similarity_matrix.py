"""CLI command: matchms similarity matrix.

Computes a similarity matrix between the spectra of one or two files and
writes it to an output artifact. Stdout carries only a summary (JSON or
human-readable).

Rows correspond to ``SPECTRA_1`` and columns to ``SPECTRA_2``. When
``SPECTRA_2`` is omitted, the result is a symmetric all-vs-all matrix over
``SPECTRA_1``. The output format is chosen by the ``-o`` extension: ``.npz``
stores the :class:`~matchms.scores.Scores` artifact, ``.tsv``/``.csv`` store
a long format with one row per pair (``row``, ``col``, optional
``row_id``/``col_id`` and one column per score field). An existing output
file is replaced.
"""

import logging
import os
import time
import numpy as np
import pandas as pd
from matchms.cli import files, similarity
from matchms.cli.constants import (
    ERROR_COMPUTE_ERROR,
    ERROR_EMPTY_SPECTRA,
    ERROR_INVALID_PARAMETER,
    ERROR_MATRIX_TOO_LARGE,
    ERROR_SAVE_FAILED,
    ERROR_UNSUPPORTED_METHOD,
)
from matchms.cli.errors import CliError
from matchms.cli.output import human_table
from matchms.cli.params import describe_signature
from matchms.scores import Scores
from matchms.similarity import __all__ as SIMILARITY_NAMES
from matchms.similarity import get_similarity_function_by_name
from matchms.similarity.base_similarity import BaseSimilarityWithSparse


CLI_COMMAND = "similarity matrix"

OUTPUT_FORMATS = ("csv", "npz", "tsv")

# Long TSV/CSV output has one row per pair, so a dense result grows with
# n*m rows; above this a sparse run with --score-min is the right tool.
DENSE_TSV_MAX_ENTRIES = 1_000_000

DEFAULT_MAX_DENSE_ENTRIES = 50_000_000

# Metadata field FingerprintSimilarity needs (matchms.fingerprints keys off
# the InChIKey to derive a spectrum's fingerprint).
_FINGERPRINT_STRUCTURE_KEY = "inchikey"

logger = logging.getLogger("matchms.cli")


def _extension(path: str) -> str | None:
    """Kept for importers; delegates to the shared helper."""
    return files.extension_of(path)


def _resolve_method(name: str, operation: str) -> tuple[str, type]:
    """Resolve --method to (canonical name, class); all similarities are valid."""
    return similarity.resolve_method(
        name,
        operation,
        valid=sorted(SIMILARITY_NAMES),
        kind="similarity method",
    )


def _sparse_capable_names() -> list[str]:
    """Names of the similarity classes that implement ``sparse_matrix``."""
    return sorted(
        name for name in SIMILARITY_NAMES if issubclass(get_similarity_function_by_name(name), BaseSimilarityWithSparse)
    )


def _validate_args(args, operation: str) -> str:
    """Check flag combinations and the output extension before loading files.

    Returns the resolved output format.
    """
    if args.score_min is not None and args.mode != "sparse":
        raise CliError(
            "--score-min is only allowed with --mode sparse.",
            code=ERROR_INVALID_PARAMETER,
            operation=operation,
            parameter="score_min",
            valid_values=["--mode sparse"],
            hint="Use --mode sparse --score-min VALUE to keep only pairs with score >= VALUE.",
        )
    if args.top < 0:
        raise CliError(
            f"--top must be a non-negative integer, got {args.top}.",
            code=ERROR_INVALID_PARAMETER,
            operation=operation,
            parameter="top",
        )
    return files.validate_output_format(
        args.output,
        operation,
        valid_formats=OUTPUT_FORMATS,
        hint="Use .npz (Scores artifact) or .tsv/.csv (long format). An existing output file is replaced.",
    )


def _validate_inputs(args, operation: str) -> list[tuple[str, str]]:
    """Check the input files exist with supported extensions and return the
    (path, format) pairs given (format is detected from the extension only).
    """
    specs = []
    for path in (args.spectra_1, args.spectra_2):
        if path is None:
            continue
        specs.append(
            (path, files.validate_input_file(path, operation, kind="input file"))
        )
    return specs


def _check_method_params(name: str, sig: dict, params: dict, operation: str) -> None:
    """Method-specific, parameter-level checks for ``matrix`` (before any file is
    loaded): required constructor parameters must be provided, and EntropySearch
    only supports fragment matching with an absolute (Da) tolerance.

    FingerprintSimilarity's required ``fingerprint_generator`` is an RDKit
    object and cannot be passed via --param, so it is skipped here and its
    usability is instead validated by the structure-metadata check after loading.
    """
    skip = ("fingerprint_generator",) if name == "FingerprintSimilarity" else ()
    similarity.check_missing_required(name, sig, params, operation, skip=skip)

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
        if params.get("matching_mode", "fragment") != "fragment":
            raise CliError(
                "EntropySearch only supports fragment matching.",
                code=ERROR_INVALID_PARAMETER,
                operation=operation,
                parameter="matching_mode",
                valid_values=["fragment"],
                hint="Pass --param matching_mode=fragment, or use `--method Entropy` for other modes.",
            )


def _check_sparse_mode(args, name: str, cls, operation: str) -> None:
    """--mode sparse requires a BaseSimilarityWithSparse implementation."""
    if args.mode != "sparse":
        return
    if not issubclass(cls, BaseSimilarityWithSparse):
        sparse_names = _sparse_capable_names()
        raise CliError(
            f"The similarity '{name}' does not support sparse_matrix().",
            code=ERROR_UNSUPPORTED_METHOD,
            operation=operation,
            parameter="mode",
            valid_values=sparse_names,
            hint=(
                f"These similarities support --mode sparse: {', '.join(sparse_names)}. Otherwise run with --mode dense."
            ),
        )


def _check_structure_metadata(cls, name: str, collections: list, operation: str) -> None:
    """FingerprintSimilarity needs InChIKey metadata in the spectra it scores."""
    if cls.__name__ != "FingerprintSimilarity":
        return
    for path, collection in collections:
        meta = collection.metadata
        if _FINGERPRINT_STRUCTURE_KEY not in meta.columns or not meta[_FINGERPRINT_STRUCTURE_KEY].notna().any():
            raise CliError(
                f"The similarity '{name}' requires '{_FINGERPRINT_STRUCTURE_KEY}' metadata, "
                f"but no spectrum in {path} provides one.",
                code=ERROR_INVALID_PARAMETER,
                operation=operation,
                input_file=path,
                parameter=_FINGERPRINT_STRUCTURE_KEY,
                hint=(
                    "Add InChIKey metadata to the input spectra (e.g. with a filter pipeline) "
                    "before computing fingerprint similarity."
                ),
            )


def _check_dense_size(n_rows: int, n_cols: int, max_entries: int, operation: str) -> None:
    if n_rows * n_cols > max_entries:
        raise CliError(
            f"The dense result would have {n_rows} x {n_cols} = {n_rows * n_cols} entries, "
            f"which exceeds --max-dense-entries ({max_entries}).",
            code=ERROR_MATRIX_TOO_LARGE,
            operation=operation,
            parameter="max_dense_entries",
            hint="Use --mode sparse --score-min VALUE to keep only the relevant pairs, or raise --max-dense-entries.",
        )


def _score_filter_for(score_min: float):
    """--score-min applies to the main 'score' field (not the auxiliary fields)."""

    def _keep(score) -> bool:
        return bool(score["score"] >= score_min)

    return _keep


def _compute(args, similarity, name: str, collection_1, collection_2, operation: str) -> Scores:
    progress_bar = not args.no_progress
    try:
        if args.mode == "sparse":
            return similarity.sparse_matrix(
                collection_1,
                collection_2,
                score_filter=_score_filter_for(args.score_min) if args.score_min is not None else None,
                progress_bar=progress_bar,
            )
        return similarity.matrix(collection_1, collection_2, progress_bar=progress_bar)
    except NotImplementedError as exc:
        raise CliError(
            f"The similarity '{name}' does not support this computation mode.",
            code=ERROR_UNSUPPORTED_METHOD,
            operation=operation,
            parameter="mode",
            valid_values=["dense", "sparse"],
            hint="Run with --mode dense.",
        ) from exc
    except Exception as exc:
        raise CliError(
            f"Similarity computation failed: {exc}",
            code=ERROR_COMPUTE_ERROR,
            operation=operation,
            hint="Check the similarity method and its parameters, and the input spectra "
            "(e.g. FingerprintSimilarity needs InChIKey metadata).",
        ) from exc


def _row_ids(collection, id_field: str, path: str, operation: str) -> list[str]:
    """Human-readable identifiers per spectrum from a metadata field."""
    meta = collection.metadata
    if id_field not in meta.columns:
        raise CliError(
            f"Metadata field '{id_field}' was not found in {path}.",
            code=ERROR_INVALID_PARAMETER,
            operation=operation,
            input_file=path,
            parameter="id_field",
            valid_values=[str(c) for c in meta.columns],
        )
    values = meta[id_field].tolist()
    return ["" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v) for v in values]


def _field_stats_dense(arr: np.ndarray) -> dict:
    finite = arr[np.isfinite(arr.astype(float))].astype(float)
    if finite.size == 0:
        return {"count": int(arr.size), "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(arr.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def _field_stats_sparse(coo) -> dict:
    if coo.nnz == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    values = coo.data.astype(float)
    return {
        "count": int(coo.nnz),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _long_format_df(scores: Scores, row_ids: list[str] | None, col_ids: list[str] | None) -> pd.DataFrame:
    """Long format: one row per stored pair, columns row, col, [row_id, col_id], <fields>."""
    if scores.is_sparse:
        # Anchor the rows on the field with the most stored values; the other
        # fields are zero-filled at the coordinates they do not cover.
        anchor_field = max(scores.score_fields, key=lambda field: scores.to_coo(field).nnz)
        anchor = scores.to_coo(anchor_field)
        row = anchor.row
        col = anchor.col
        positions = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(row.tolist(), col.tolist(), strict=True))}
    else:
        n_rows, n_cols = scores.shape
        row = np.repeat(np.arange(n_rows), n_cols)
        col = np.tile(np.arange(n_cols), n_rows)
        positions = None

    df = pd.DataFrame({"row": row, "col": col})
    if row_ids is not None:
        df["row_id"] = pd.Series(row_ids).take(row).to_numpy()
        df["col_id"] = pd.Series(col_ids).take(col).to_numpy()
    for field in scores.score_fields:
        if scores.is_sparse:
            values = np.zeros(len(row), dtype=scores.to_coo(field).dtype)
            field_coo = scores.to_coo(field)
            for r, c, v in zip(field_coo.row.tolist(), field_coo.col.tolist(), field_coo.data.tolist(), strict=True):
                i = positions.get((r, c))
                if i is not None:
                    values[i] = v
        else:
            values = scores.to_array(field).ravel()
        df[field] = values
    return df


def _save_scores(
    args, scores: Scores, output_format: str, row_ids, col_ids, n_rows: int, n_cols: int, operation: str
) -> str:
    if output_format == "npz":
        try:
            scores.save(args.output)
        except Exception as exc:
            raise CliError(
                f"Failed to save the scores to {args.output}: {exc}",
                code=ERROR_SAVE_FAILED,
                operation=operation,
                input_file=args.output,
            ) from exc
        return output_format

    if args.mode == "dense" and n_rows * n_cols > DENSE_TSV_MAX_ENTRIES:
        raise CliError(
            f"Writing a dense result of {n_rows} x {n_cols} pairs to .{output_format} would produce "
            f"{n_rows * n_cols} rows (limit {DENSE_TSV_MAX_ENTRIES}).",
            code=ERROR_MATRIX_TOO_LARGE,
            operation=operation,
            parameter="mode",
            hint="Use --mode sparse --score-min VALUE to write only the kept pairs, "
            "or save to .npz for the full dense result.",
        )
    df = _long_format_df(scores, row_ids, col_ids)
    delimiter = "," if output_format == "csv" else "\t"
    try:
        df.to_csv(args.output, sep=delimiter, index=False)
    except Exception as exc:
        raise CliError(
            f"Failed to save the scores to {args.output}: {exc}",
            code=ERROR_SAVE_FAILED,
            operation=operation,
            input_file=args.output,
        ) from exc
    return output_format


def _top_pairs(
    scores: Scores, symmetric: bool, top: int, row_ids: list[str] | None, col_ids: list[str] | None
) -> list[dict]:
    """The top *top* pairs of the main score field (diagonal excluded when symmetric)."""
    main_field = "score" if "score" in scores.score_fields else scores.score_fields[0]
    if scores.is_sparse:
        coo = scores.to_coo(main_field)
        row, col, value = coo.row, coo.col, coo.data
        if symmetric:
            keep = row != col
            row, col, value = row[keep], col[keep], value[keep]
    else:
        arr = scores.to_array(main_field)
        if symmetric:
            row, col = np.triu_indices(arr.shape[0], k=1)
            value = arr[row, col]
        else:
            n_rows, n_cols = arr.shape
            row = np.repeat(np.arange(n_rows), n_cols)
            col = np.tile(np.arange(n_cols), n_rows)
            value = arr.ravel()
    if top == 0 or value.size == 0:
        return []
    # Cast to float: some score fields are boolean (e.g. PrecursorMzMatch),
    # and unary minus is not defined on numpy booleans.
    order = np.argsort(-value.astype(float))[:top]
    pairs = []
    for i in order:
        pair = {
            "row": int(row[i]),
            "col": int(col[i]),
        }
        if row_ids is not None:
            pair["row_id"] = row_ids[int(row[i])]
            pair["col_id"] = col_ids[int(col[i])]
        pair["value"] = float(value[i])
        pairs.append(pair)
    return pairs


def run(args, ctx) -> int:
    """Run the `similarity matrix` command."""
    operation = CLI_COMMAND
    output_format = _validate_args(args, operation)
    input_specs = _validate_inputs(args, operation)

    method_name, cls = _resolve_method(args.method, operation)
    params = similarity.build_params(args, operation)
    signature = describe_signature(cls)
    similarity.check_params_accepted(cls, method_name, params, operation)
    effective = similarity.effective_params(signature, params)
    _check_method_params(method_name, signature, params, operation)
    _check_sparse_mode(args, method_name, cls, operation)

    collection_1 = files.load_collection(input_specs[0][0], input_specs[0][1], operation)
    n_rows = len(collection_1)
    if n_rows == 0:
        raise CliError(
            f"No spectra were loaded from {input_specs[0][0]}; nothing to compare.",
            code=ERROR_EMPTY_SPECTRA,
            operation=operation,
            input_file=input_specs[0][0],
        )

    symmetric = args.spectra_2 is None
    if not symmetric:
        if os.path.abspath(input_specs[0][0]) == os.path.abspath(input_specs[1][0]):
            logger.warning(
                "SPECTRA_1 and SPECTRA_2 are the same file; running a two-file (non-symmetric) "
                "computation. Omit SPECTRA_2 for a symmetric all-vs-all run."
            )
        collection_2 = files.load_collection(input_specs[1][0], input_specs[1][1], operation)
        n_cols = len(collection_2)
        if n_cols == 0:
            raise CliError(
                f"No spectra were loaded from {input_specs[1][0]}; nothing to compare.",
                code=ERROR_EMPTY_SPECTRA,
                operation=operation,
                input_file=input_specs[1][0],
            )
        collections = [(input_specs[0][0], collection_1), (input_specs[1][0], collection_2)]
    else:
        collection_2 = collection_1
        n_cols = n_rows
        collections = [(input_specs[0][0], collection_1)]

    _check_structure_metadata(cls, method_name, collections, operation)
    if args.mode == "dense":
        _check_dense_size(n_rows, n_cols, args.max_dense_entries, operation)

    similarity_instance = similarity.instantiate(cls, method_name, params, operation, fingerprint_hint=True)

    row_ids = _row_ids(collection_1, args.id_field, input_specs[0][0], operation) if args.id_field else None
    if row_ids is not None and not symmetric:
        col_ids = _row_ids(collection_2, args.id_field, input_specs[1][0], operation)
    else:
        col_ids = row_ids

    started = time.perf_counter()
    scores = _compute(args, similarity_instance, method_name, collection_1, collection_2, operation)
    elapsed = time.perf_counter() - started

    format_name = _save_scores(
        args,
        scores,
        output_format,
        row_ids=row_ids,
        col_ids=col_ids,
        n_rows=n_rows,
        n_cols=n_cols,
        operation=operation,
    )

    main_field = "score" if "score" in scores.score_fields else scores.score_fields[0]
    if scores.is_sparse:
        n_stored = int(scores.to_coo(main_field).nnz)
    else:
        n_stored = n_rows * n_cols

    stats = {}
    for field in scores.score_fields:
        if scores.is_sparse:
            stats[field] = _field_stats_sparse(scores.to_coo(field))
        else:
            stats[field] = _field_stats_dense(scores.to_array(field))

    payload = {
        "ok": True,
        "operation": CLI_COMMAND,
        "method": {
            "name": method_name,
            "class": cls.__name__,
            "params": effective,
        },
        "mode": args.mode,
        "inputs": {
            "spectra_1": {"file": input_specs[0][0], "n_spectra": n_rows},
            "spectra_2": ({"file": input_specs[1][0], "n_spectra": n_cols} if not symmetric else None),
            "symmetric": symmetric,
        },
        "scores": {
            "shape": [n_rows, n_cols],
            "kind": "sparse" if scores.is_sparse else "dense",
            "score_fields": list(scores.score_fields),
            "stats": stats,
            "n_stored": n_stored,
            "density": n_stored / (n_rows * n_cols),
        },
        "top_pairs": _top_pairs(scores, symmetric, args.top, row_ids, col_ids),
        "elapsed_seconds": round(elapsed, 6),
        "output": {
            "file": args.output,
            "format": format_name,
            "size_bytes": os.path.getsize(args.output),
        },
    }

    if ctx.machine_mode:
        ctx.write_json(payload)
    else:
        ctx.write_text(_format_human(payload, args))
    return 0


def _format_human(payload: dict, args) -> str:
    scores = payload["scores"]
    inputs = payload["inputs"]
    lines = [
        f"Similarity matrix: {inputs['spectra_1']['file']}"
        + (f" x {inputs['spectra_2']['file']}" if inputs["spectra_2"] else " (symmetric)"),
        f"  method:   {payload['method']['name']} "
        f"({', '.join(f'{k}={v}' for k, v in payload['method']['params'].items())})",
        f"  mode:     {payload['mode']} (rows x columns = {scores['shape'][0]} x {scores['shape'][1]})",
        f"  stored:   {scores['n_stored']} pairs (density {scores['density']:.4g})",
    ]
    if args.score_min is not None:
        lines.append(f"  filter:   score >= {args.score_min} (applied to the main 'score' field)")
    for field, stat in scores["stats"].items():
        if stat["min"] is None:
            lines.append(f"  field {field}: no stored values")
        else:
            lines.append(
                f"  field {field}: min {stat['min']:.6g}, max {stat['max']:.6g}, "
                f"mean {stat['mean']:.6g}, std {stat['std']:.6g} (n={stat['count']})"
            )
    lines.append(f"  elapsed:  {payload['elapsed_seconds']:.3f} s")
    lines.append("")

    if payload["top_pairs"]:
        id_field = args.id_field
        columns = ["row", "col"] + (["row_id", "col_id"] if id_field else []) + ["value"]
        lines.append(
            f"Top {len(payload['top_pairs'])} pairs (main 'score' field"
            + (", symmetric diagonal excluded" if inputs["symmetric"] else "")
            + "):"
        )
        lines.append(
            human_table(
                columns,
                [[pair[c] for c in columns] for pair in payload["top_pairs"]],
            )
        )
        lines.append("")

    out = payload["output"]
    lines.append(f"Saved to {out['file']} ({out['format']}, {out['size_bytes']} bytes).")
    if not inputs["symmetric"] and os.path.abspath(inputs["spectra_1"]["file"]) == os.path.abspath(
        inputs["spectra_2"]["file"]
    ):
        lines.append("Note: both inputs are the same file; a two-file (non-symmetric) run was performed.")
    lines.append("")
    lines.append("Run with --json for the machine-readable version.")
    return "\n".join(lines)
