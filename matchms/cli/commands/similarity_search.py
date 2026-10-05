"""CLI command: matchms similarity search.

Searches the spectra of a query file against a reference library and writes the
best matches per query to a hit list. Stdout carries only a summary (JSON or
human-readable); the hit list itself goes to the ``-o`` file.

The library is either a spectra file (the index is built on the fly) or an index
saved by ``matchms similarity build-index`` (recognized by its ``.index.npz``
extension). As with the Python ``similarity.search`` API, the queries are always
the rows of the score matrix and the library is always the reference set
(columns); a query may be its own best hit (there is no self-match exclusion).

``reference_index`` always refers to the row order of the original library file.
For an index this means the file it was built from must be kept alongside it
(pass ``--library-spectra``) when readable identifiers are required.
"""

import os
import time
import numpy as np
import pandas as pd
from matchms.cli.commands.similarity_build_index import (
    _build_params,
    _check_index_capable,
    _check_method_params,
    _check_params_accepted,
    _effective_params,
    _extension,
    _instantiate,
    _load_collection,
    _resolve_method,
)
from matchms.cli.errors import CliError, raise_for_unknown_value
from matchms.importing.load_spectra import SUPPORTED_FILE_FORMATS as INPUT_FORMATS


CLI_COMMAND = "similarity search"

OUTPUT_FORMATS = ("csv", "tsv")

INDEX_SUFFIX = ".index.npz"


def _is_index_path(path: str) -> bool:
    return path.lower().endswith(INDEX_SUFFIX)


def _validate_output(args, operation: str) -> str:
    """The hit list must be a .tsv or .csv file whose directory exists and is writable."""
    output = args.output
    output_format = _extension(output)
    if output_format not in OUTPUT_FORMATS:
        raise CliError(
            f"Output file '{output}' must end in '.tsv' or '.csv'.",
            code="invalid_parameter",
            operation=operation,
            parameter="output",
            valid_values=[f".{name}" for name in OUTPUT_FORMATS],
            hint="The hit list is written in a long table format (.tsv or .csv).",
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
    return output_format


def _param_names(pairs: list[str] | None) -> list[str]:
    names = []
    for pair in pairs or []:
        if "=" in pair:
            names.append(pair.partition("=")[0].strip())
    return names


def _validate_args(args, operation: str) -> str:
    """Validate the flags before any file is read; returns the output format."""
    output_format = _validate_output(args, operation)

    if args.top_k is None or args.top_k < 1:
        raise CliError(
            f"--top-k must be a positive integer (>= 1), got {args.top_k}.",
            code="invalid_parameter",
            operation=operation,
            parameter="top_k",
            valid_values=["1 or higher"],
        )
    if args.batch_size is None or args.batch_size < 1:
        raise CliError(
            f"--batch-size must be a positive integer (>= 1), got {args.batch_size}.",
            code="invalid_parameter",
            operation=operation,
            parameter="batch_size",
            valid_values=["1 or higher"],
        )
    if args.tolerance is not None and "tolerance" in _param_names(args.param):
        raise CliError(
            "--tolerance and --param tolerance=... may not be combined.",
            code="invalid_parameter",
            operation=operation,
            parameter="tolerance",
            hint="Use either the --tolerance shorthand or --param tolerance=..., not both.",
        )

    library_is_index = _is_index_path(args.library)
    if args.library_spectra is not None and not library_is_index:
        raise CliError(
            "--library-spectra is only allowed when LIBRARY is an index (.index.npz). "
            "A spectra library already carries its own metadata, so no separate "
            "identifier file is needed.",
            code="invalid_parameter",
            operation=operation,
            parameter="library_spectra",
            hint="Remove --library-spectra, or pass an .index.npz file as LIBRARY.",
        )
    if args.library_id_field is not None and library_is_index and args.library_spectra is None:
        raise CliError(
            "--library-id-field requires --library-spectra when LIBRARY is an index, "
            "because an index does not store spectrum metadata.",
            code="invalid_parameter",
            operation=operation,
            parameter="library_id_field",
            hint="Pass --library-spectra with the spectra file the index was built from.",
        )
    return output_format


def _validate_library_path(args, operation: str) -> str:
    """Validate the LIBRARY argument; returns 'index' or 'spectra'."""
    path = args.library
    if not os.path.exists(path):
        raise CliError(
            f"The specified library file: {path} does not exist.",
            code="input_not_found",
            operation=operation,
            input_file=path,
            valid_values=[INDEX_SUFFIX] + sorted(INPUT_FORMATS),
            hint="Expected a reference library: a spectra file with a supported extension "
            "(e.g. .mgf, .msp, .mzml, .mzxml, .json, .pickle) or an index ending in "
            ".index.npz (from `similarity build-index`).",
        )
    if _is_index_path(path):
        return "index"
    file_format = _extension(path)
    if file_format not in INPUT_FORMATS:
        raise CliError(
            f"Library file extension '.{file_format}' of {path} is not a supported input format. "
            "The input format is detected from the file extension only, so files with a "
            "non-standard extension cannot be loaded.",
            code="unsupported_format",
            operation=operation,
            input_file=path,
            valid_values=sorted(INPUT_FORMATS),
            hint="Use a supported extension such as .mgf, .msp, .mzml, .mzxml, .json or .pickle, "
            "or pass a .index.npz index built by `similarity build-index`.",
        )
    return "spectra"


def _validate_query_path(args, operation: str) -> str:
    """Validate the QUERIES argument and return its file format."""
    path = args.queries
    if not os.path.exists(path):
        raise CliError(
            f"The specified query file: {path} does not exist.",
            code="input_not_found",
            operation=operation,
            input_file=path,
            valid_values=sorted(INPUT_FORMATS),
            hint="Expected a query spectra file with a supported extension "
            "(e.g. .mgf, .msp, .mzml, .mzxml, .json, .pickle).",
        )
    file_format = _extension(path)
    if file_format not in INPUT_FORMATS:
        raise CliError(
            f"Query file extension '.{file_format}' of {path} is not a supported input format. "
            "The input format is detected from the file extension only, so files with a "
            "non-standard extension cannot be loaded.",
            code="unsupported_format",
            operation=operation,
            input_file=path,
            valid_values=sorted(INPUT_FORMATS),
            hint="Use a supported extension such as .mgf, .msp, .mzml, .mzxml, .json or .pickle.",
        )
    return file_format


def _validate_library_spectra_path(args, operation: str) -> str:
    """Validate --library-spectra (only present when LIBRARY is an index)."""
    path = args.library_spectra
    if path is None:
        raise CliError(
            "--library-spectra is required for this check but was not provided.",
            code="invalid_parameter",
            operation=operation,
            parameter="library_spectra",
        )
    if not os.path.exists(path):
        raise CliError(
            f"The specified --library-spectra file: {path} does not exist.",
            code="input_not_found",
            operation=operation,
            input_file=path,
            valid_values=sorted(INPUT_FORMATS),
            hint="Expected the spectra file the index was built from (a supported spectra extension).",
        )
    file_format = _extension(path)
    if file_format not in INPUT_FORMATS:
        raise CliError(
            f"--library-spectra file extension '.{file_format}' of {path} is not a supported input format.",
            code="unsupported_format",
            operation=operation,
            input_file=path,
            valid_values=sorted(INPUT_FORMATS),
            hint="Use a supported extension such as .mgf, .msp, .mzml, .mzxml, .json or .pickle.",
        )
    return file_format


def _check_score_field(cls, name: str, field: str, operation: str) -> None:
    """--score-field must be one of the method's score fields."""
    valid = list(cls.score_fields)
    if field in valid:
        return
    raise_for_unknown_value(
        operation=operation,
        parameter="score_field",
        value=field,
        valid=valid,
        kind=f"score field for the similarity '{name}'",
    )


def _load_index(similarity, name: str, path: str, operation: str):
    """Load a saved index, translating a configuration mismatch to index_incompatible."""
    try:
        return similarity.load_index(path)
    except (ModuleNotFoundError, ImportError) as exc:
        raise CliError(
            f"A required dependency of the similarity '{name}' is not installed: {exc}",
            code="missing_dependency",
            operation=operation,
            parameter=name,
            hint="Install the missing dependency and retry.",
        ) from exc
    except ValueError as exc:
        raise _index_incompatible(name, path, str(exc), operation) from exc
    except Exception as exc:
        raise CliError(
            f"Failed to read the index file {path}: {exc}",
            code="compute_error",
            operation=operation,
            input_file=path,
            hint="Check that the file is a valid index built by `similarity build-index` with this matchms version.",
        ) from exc


def _build_index(similarity, name: str, collection, operation: str):
    """Build an index from a spectra library, mapping failure modes to structured errors."""
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
        if "use peak_separation='merge'" in message:
            raise CliError(
                f"The library violates the peak-separation requirement of '{name}' (peak_separation=raise): {message}",
                code="invalid_input",
                operation=operation,
                hint="Rebuild with --param peak_separation=merge (close peaks are merged) "
                "or preprocess the library so no two peaks are closer than 2 * max_tolerance.",
            ) from exc
        raise _index_incompatible(name, "<built from the library spectra>", message, operation) from exc
    except Exception as exc:
        raise _index_incompatible(name, "<built from the library spectra>", str(exc), operation) from exc


def _index_incompatible(name: str, source: str, message: str, operation: str) -> CliError:
    differing = _extract_differing(message)
    if differing:
        message = f"{message} Differing parameters: {', '.join(differing)}."
    return CliError(
        f"The library index {source} does not match the similarity '{name}' configuration: {message}",
        code="index_incompatible",
        operation=operation,
        hint="The index was prepared with different --method/--param settings. Rebuild the "
        "index with `similarity build-index` using the same values, or pass the matching "
        "parameters to `similarity search`.",
    )


def _extract_differing(message: str) -> list[str]:
    marker = "differs for:"
    if marker not in message:
        return []
    tail = message.split(marker, 1)[1].split("Rebuild")[0]
    return [part.strip() for part in tail.split(",") if part.strip()]


def _search_batch(similarity, name: str, batch, library_index, batch_no: int, operation: str):
    try:
        return similarity.search(batch, library_index, progress_bar=False, n_jobs=1)
    except (ModuleNotFoundError, ImportError) as exc:
        raise CliError(
            f"A required dependency of the similarity '{name}' is not installed: {exc}",
            code="missing_dependency",
            operation=operation,
            parameter=name,
            hint="Install the missing dependency and retry.",
        ) from exc
    except ValueError as exc:
        raise CliError(
            f"The library index does not match the similarity '{name}' configuration (batch {batch_no}): {exc}",
            code="index_incompatible",
            operation=operation,
            hint="The index and the search parameters must use the same preprocessing settings.",
        ) from exc
    except Exception as exc:
        raise CliError(
            f"Similarity search failed (batch {batch_no}): {exc}",
            code="compute_error",
            operation=operation,
            hint="Check the similarity method and its parameters, and the input spectra.",
        ) from exc


def _search_in_batches(
    args,
    similarity,
    name: str,
    library_index,
    queries,
    score_fields: tuple[str, ...],
    operation: str,
) -> list[dict]:
    """Search in batches; keep the top-k hits per query ranked by --score-field.

    Returns one record per kept hit: {query_index, reference_index, rank, <field>...}.
    Records are ordered by query index, then rank.
    """
    top_k = args.top_k
    min_score = args.min_score
    score_field = args.score_field
    batch_size = args.batch_size
    n_queries = len(queries)

    hits: list[dict] = []
    for start in range(0, n_queries, batch_size):
        batch_no = start // batch_size + 1
        batch = queries[start : start + batch_size]
        scores = _search_batch(similarity, name, batch, library_index, batch_no, operation)
        row_values = {field: scores.to_array(field) for field in score_fields}
        for offset in range(row_values[score_field].shape[0]):
            ranking = row_values[score_field][offset]
            order = np.argsort(-ranking, kind="stable")
            for rank, ref in enumerate(order, start=1):
                if rank > top_k:
                    break
                value = float(ranking[ref])
                if min_score is not None and value < min_score:
                    break  # ranking is descending, so the rest are below the threshold
                record = {
                    "query_index": int(start + offset),
                    "reference_index": int(ref),
                    "rank": int(rank),
                }
                for field in score_fields:
                    record[field] = _to_python(row_values[field][offset, ref])
                hits.append(record)
    return hits


def _to_python(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def _read_ids(collection, field: str, kind: str, path: str, operation: str) -> list[str]:
    """Read a metadata column as a list of string ids (one per spectrum, in row order)."""
    meta = collection.metadata
    if field not in meta.columns:
        raise CliError(
            f"Metadata field '{field}' was not found in the {kind} file {path}.",
            code="invalid_parameter",
            operation=operation,
            input_file=path,
            parameter="query_id_field" if kind == "query" else "library_id_field",
            valid_values=[str(column) for column in meta.columns],
            hint=f"Pass a metadata field present in the {kind} file (see `matchms spectra describe`), "
            "or omit the --...-id-field flag to omit the column.",
        )
    ids = []
    for value in meta[field].tolist():
        if _is_missing(value):
            ids.append("")
        else:
            ids.append(str(value))
    return ids


def _is_missing(value) -> bool:
    """True for None / NaN / NaT and other pandas NA values (scalar cells)."""
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _write_hit_list(
    args,
    hits: list[dict],
    query_ids: list[str] | None,
    library_ids: list[str] | None,
    score_fields: tuple[str, ...],
    output_format: str,
    operation: str,
) -> int:
    columns = ["query_index", "reference_index", "rank"]
    if args.query_id_field is not None:
        columns.insert(1, "query_id")
    if args.library_id_field is not None:
        columns.insert(3, "reference_id")
    columns = columns + list(score_fields)

    records: list[dict] = []
    for hit in hits:
        record = {"query_index": hit["query_index"], "reference_index": hit["reference_index"], "rank": hit["rank"]}
        if args.query_id_field is not None:
            record["query_id"] = query_ids[hit["query_index"]]
        if args.library_id_field is not None:
            record["reference_id"] = library_ids[hit["reference_index"]]
        for field in score_fields:
            record[field] = hit[field]
        records.append(record)

    frame = pd.DataFrame(records, columns=columns)
    delimiter = "," if output_format == "csv" else "\t"
    try:
        frame.to_csv(args.output, sep=delimiter, index=False)
    except Exception as exc:
        raise CliError(
            f"Failed to write the hit list to {args.output}: {exc}",
            code="save_failed",
            operation=operation,
            input_file=args.output,
        ) from exc
    return int(frame.shape[0])


def _top_hits(
    hits: list[dict],
    query_ids: list[str] | None,
    library_ids: list[str] | None,
    top: int,
    score_field: str,
) -> list[dict]:
    """The top hits over all queries, ranked by the score field (ties in hit order)."""
    if top <= 0 or not hits:
        return []
    decorated = [(-float(hit[score_field]), position, hit) for position, hit in enumerate(hits)]
    decorated.sort(key=lambda item: (item[0], item[1]))
    result = []
    for negative_value, _position, hit in decorated[:top]:
        entry: dict = {
            "query_index": hit["query_index"],
            "reference_index": hit["reference_index"],
            "value": -negative_value,
        }
        if query_ids is not None:
            entry["query_id"] = query_ids[hit["query_index"]]
        if library_ids is not None:
            entry["reference_id"] = library_ids[hit["reference_index"]]
        result.append(entry)
    return result


def _prepare_library(args, similarity, method_name: str, library_kind: str, operation: str) -> tuple:
    """Load or build the library index.

    Returns (library_index, n_spectra, meta_source, spectra_file) where meta_source
    is the SpectraCollection whose row order maps to reference_index (for ids).
    """
    if library_kind == "index":
        index = _load_index(similarity, method_name, args.library, operation)
        n_spectra = int(index.n_specs)
        meta_source = None
        spectra_file = None
        if args.library_spectra is not None:
            meta_source = _load_collection(
                args.library_spectra,
                _validate_library_spectra_path(args, operation),
                operation,
            )
            if len(meta_source) != n_spectra:
                raise CliError(
                    f"--library-spectra {args.library_spectra} contains {len(meta_source)} spectra, "
                    f"but the index contains {n_spectra}. They must match.",
                    code="invalid_input",
                    operation=operation,
                    input_file=args.library_spectra,
                    valid_values=[f"{n_spectra} spectra"],
                    hint="Pass the spectra file the index was built from (same number of spectra, in the same order).",
                )
            spectra_file = args.library_spectra
        return index, n_spectra, meta_source, spectra_file

    collection = _load_collection(args.library, _extension(args.library), operation)
    index = _build_index(similarity, method_name, collection, operation)
    return index, len(collection), collection, args.library


def run(args, ctx) -> int:
    """Run the `similarity search` command."""
    operation = CLI_COMMAND

    # 1. Validate arguments (no file is read).
    output_format = _validate_args(args, operation)
    method_name, cls = _resolve_method(args.method, operation)
    _check_index_capable(cls, method_name, operation)
    params = _build_params(args, operation)
    _check_params_accepted(cls, method_name, params, operation)
    _check_method_params(method_name, _effective_params(cls, params), operation)
    similarity = _instantiate(cls, method_name, params, operation)
    _check_score_field(cls, method_name, args.score_field, operation)

    library_kind = _validate_library_path(args, operation)
    query_format = _validate_query_path(args, operation)
    if args.library_spectra is not None:
        _validate_library_spectra_path(args, operation)

    # 2. Prepare the library (load a saved index or build one on the fly).
    library_index, library_n, library_meta, spectra_file = _prepare_library(
        args,
        similarity,
        method_name,
        library_kind,
        operation,
    )

    # 3. Load the queries.
    query_collection = _load_collection(args.queries, query_format, operation)
    n_queries = len(query_collection)

    # 4. Resolve the optional identifier columns.
    query_ids = (
        _read_ids(query_collection, args.query_id_field, "query", args.queries, operation)
        if args.query_id_field is not None
        else None
    )
    library_ids = None
    if args.library_id_field is not None:
        library_ids = _read_ids(
            library_meta,
            args.library_id_field,
            "library",
            spectra_file if library_kind == "index" else args.library,
            operation,
        )

    # 5. Search in batches.
    started = time.perf_counter()
    hits = _search_in_batches(
        args,
        similarity,
        method_name,
        library_index,
        query_collection,
        tuple(cls.score_fields),
        operation,
    )
    elapsed = time.perf_counter() - started

    # 6. Write the hit list.
    n_rows = _write_hit_list(
        args,
        hits,
        query_ids,
        library_ids,
        tuple(cls.score_fields),
        output_format,
        operation,
    )
    n_with_hits = len({hit["query_index"] for hit in hits})
    top = _top_hits(hits, query_ids, library_ids, args.top, args.score_field)

    payload = {
        "ok": True,
        "operation": CLI_COMMAND,
        "method": {
            "name": method_name,
            "class": cls.__name__,
            "params": _effective_params(cls, params),
        },
        "queries": {"file": args.queries, "n_spectra": n_queries},
        "library": {
            "file": args.library,
            "kind": library_kind,
            "n_spectra": library_n,
            "spectra_file": spectra_file,
        },
        "search": {
            "top_k": args.top_k,
            "min_score": args.min_score,
            "score_field": args.score_field,
        },
        "results": {
            "n_hits": n_rows,
            "n_queries_with_hits": n_with_hits,
            "n_queries_without_hits": n_queries - n_with_hits,
        },
        "top_hits": top,
        "elapsed_seconds": round(elapsed, 6),
        "output": {
            "file": args.output,
            "format": output_format,
            "size_bytes": os.path.getsize(args.output),
        },
    }

    if ctx.machine_mode:
        ctx.write_json(payload)
    else:
        ctx.write_text(_format_human(payload))
    return 0


def _format_human(payload: dict) -> str:
    method = payload["method"]
    library = payload["library"]
    search = payload["search"]
    results = payload["results"]
    lines = [
        f"Similarity search: {payload['queries']['n_spectra']} queries "
        f"vs {library['n_spectra']} references ({library['kind']}: {library['file']})",
        f"  method:   {method['name']} ({', '.join(f'{k}={v}' for k, v in method['params'].items())})",
        f"  top-k:    {search['top_k']}",
    ]
    if search["min_score"] is not None:
        lines.append(f"  min-score: {search['min_score']} (on field '{search['score_field']}')")
    lines.append(
        f"  hits:     {results['n_hits']} "
        f"(queries with hits: {results['n_queries_with_hits']}, "
        f"without: {results['n_queries_without_hits']})",
    )
    if payload["top_hits"]:
        lines.append("  top hits:")
        for hit in payload["top_hits"]:
            lines.append(
                f"    query {hit['query_index']} -> reference {hit['reference_index']} ({hit['value']:.6g})",
            )
    lines.append(
        f"  hit list: {payload['output']['file']} "
        f"({payload['output']['format']}, {payload['output']['size_bytes']} bytes)",
    )
    lines.append(f"  elapsed:  {payload['elapsed_seconds']:.3f} s")
    lines.append("")
    lines.append("The full hit list is in the output file. Run with --json for the machine-readable summary.")
    return "\n".join(lines)
