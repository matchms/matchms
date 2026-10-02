"""CLI command: matchms filter run.

Runs a filter pipeline over a :class:`~matchms.SpectraCollection` using
:class:`~matchms.filtering.SpectraProcessor` and writes the filtered spectra
back to disk with :func:`~matchms.exporting.save_spectra`.

The base pipeline defaults to ``DEFAULT_FILTERS``
(:mod:`matchms.filtering.default_pipelines`). A named default pipeline
(e.g. ``BASIC_FILTERS``) can be selected with ``--pipeline``, and additional
filters can be appended with ``--filter`` (they are inserted at the correct
position of the canonical matchms filter order).

Filter parameters are passed with ``--param filter_name.param=value``. The
filter named in the parameter must be part of the pipeline: either one of the
``--filter`` filters or a filter already contained in the base pipeline. The
value is typed (``true``/``false`` -> bool, numbers -> int/float, ``[...]`` /
``{...}`` -> JSON list/dict) and bound to that filter.

The processing report is optional (``--report``) and is saved as a JSON
artifact named ``<output file name>_processing_report.json`` next to the
output spectra file.
"""

import json
import os
import pandas as pd
from matchms import SpectraProcessor
from matchms.cli.errors import CliError, raise_for_unknown_value
from matchms.cli.introspection import filter_accepts_parameter, filter_parameter_names
from matchms.cli.output import human_table
from matchms.cli.params import parse_scoped_params
from matchms.exporting import save_spectra
from matchms.exporting.save_spectra import SUPPORTED_FILE_FORMATS as OUTPUT_FORMATS
from matchms.filtering.default_pipelines import DEFAULT_FILTERS, FILTER_SETS_BY_NAME
from matchms.filtering.filter_order import ALL_FILTERS, FILTER_FUNCTION_NAMES
from matchms.importing import load_ms2_dataset
from matchms.importing.load_spectra import SUPPORTED_FILE_FORMATS as INPUT_FORMATS


CLI_COMMAND = "filter run"

EXPORT_STYLES = ("matchms", "massbank", "nist", "riken", "gnps")

REPORT_SUFFIX = "_processing_report.json"

REPORT_COLUMNS = (
    ("input spectra", "input_spectra"),
    ("output spectra", "output_spectra"),
    ("removed spectra", "removed_spectra"),
    ("changed metadata", "changed_metadata"),
    ("changed fragments", "changed_fragments"),
)

PIPELINES = FILTER_SETS_BY_NAME


def _extension(path: str) -> str | None:
    return os.path.splitext(path)[1].lower().lstrip(".") or None


def _nullable_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _base_filter_params(base: list) -> dict[str, dict]:
    """Map filter name -> pipeline parameters for one default pipeline."""
    filters: dict[str, dict] = {}
    for entry in base:
        if isinstance(entry, (tuple, list)):
            filters[entry[0].__name__] = dict(entry[1])
        else:
            filters[entry.__name__] = {}
    return filters


def _build_pipeline(args, operation: str) -> tuple[str, list]:
    """Build the ordered list of filter descriptions to run.

    Returns the base pipeline name and a list of ``(name, params_or_None)``
    tuples sorted by the canonical matchms filter order.

    Raises
    ------
    CliError
        ``unknown_value`` for an unknown ``--pipeline``/``--filter`` name,
        ``unknown_parameter`` when a ``--param`` does not belong to a filter
        in the pipeline, ``invalid_parameter`` when a parameter is not
        accepted by its target filter.
    """
    if args.pipeline is not None:
        if args.pipeline not in PIPELINES:
            raise_for_unknown_value(
                operation=operation,
                parameter="pipeline",
                value=args.pipeline,
                valid=PIPELINES,
                kind="pipeline",
                hint="Use one of the default pipelines from matchms.filtering.default_pipelines (see `matchms info`).",
            )
        base_name = args.pipeline
        base = PIPELINES[base_name]
    else:
        base_name = "DEFAULT_FILTERS"
        base = DEFAULT_FILTERS

    filters = _base_filter_params(base)

    added_names: list[str] = []
    for name in args.filter or []:
        if name not in FILTER_FUNCTION_NAMES:
            raise_for_unknown_value(
                operation=operation,
                parameter="filter",
                value=name,
                valid=FILTER_FUNCTION_NAMES,
                kind="filter",
                hint="Use `matchms filter list` to see all available filter names.",
            )
        if name not in added_names:
            added_names.append(name)
        filters.setdefault(name, {})

    for target, param_map in parse_scoped_params(args.param, operation=operation).items():
        if target not in filters:
            hint = (
                f"'{target}' is not in the pipeline. Add it with "
                f"--filter {target} or choose a filter of the "
                f"'{base_name}' pipeline."
            )
            valid = sorted(set(added_names) | set(FILTER_FUNCTION_NAMES))
            raise_for_unknown_value(
                operation=operation,
                parameter="param",
                value=target,
                valid=valid,
                kind="filter (in --param filter_name.param=value)",
                hint=hint,
            )
        func = FILTER_FUNCTION_NAMES[target]
        for param_name in param_map:
            if not filter_accepts_parameter(func, param_name):
                raise CliError(
                    f"The filter '{target}' does not accept the parameter '{param_name}'.",
                    code="invalid_parameter",
                    operation=operation,
                    parameter=f"{target}.{param_name}",
                    valid_values=filter_parameter_names(func),
                    hint=f"See `matchms filter info {target}` for the available parameters.",
                )
        filters[target].update(param_map)

    ordered = sorted(
        filters.items(),
        key=lambda item: ALL_FILTERS.index(FILTER_FUNCTION_NAMES[item[0]]),
    )
    return base_name, [(name, params if params else None) for name, params in ordered]


def _make_processor(descriptions: list, operation: str) -> SpectraProcessor:
    try:
        return SpectraProcessor([name if params is None else (name, params) for name, params in descriptions])
    except AssertionError as exc:
        # A filter that still has required, unset parameters (e.g. a filter
        # added with --filter whose mandatory parameter was not passed).
        raise CliError(
            "A filter in the pipeline is missing required parameters.",
            code="missing_parameter",
            operation=operation,
            hint="Pass the missing value with --param filter_name.param=value. "
            "See `matchms filter info <name>` for the required parameters.",
        ) from exc


def _report_rows(report) -> list[dict]:
    df = report.to_dataframe()
    rows = []
    for name in df.index:
        row = {"filter": name}
        for column, key in REPORT_COLUMNS:
            row[key] = _nullable_int(df.loc[name, column])
        rows.append(row)
    return rows


def _report_path(output_file: str) -> str:
    stem, _ = os.path.splitext(output_file)
    return f"{stem}{REPORT_SUFFIX}"


def run(args, ctx) -> int:
    """Run the `filter run` command."""
    input_file = args.input
    output_file = args.output
    operation = f"{CLI_COMMAND} {input_file} -> {output_file}"

    if not os.path.exists(input_file):
        raise CliError(
            f"The specified input file: {input_file} does not exist.",
            code="file_not_found",
            operation=operation,
            input_file=input_file,
            valid_values=sorted(INPUT_FORMATS),
            hint="Expected a spectra file with a supported extension (e.g. .mgf, .msp, .mzml, .mzxml, .json, .pickle).",
        )

    input_format = _extension(input_file)
    if input_format is None or input_format not in INPUT_FORMATS:
        raise CliError(
            f"Input file extension '.{input_format}' of {input_file} is not a supported input format.",
            code="unsupported_input_format",
            operation=operation,
            input_file=input_file,
            valid_values=sorted(INPUT_FORMATS),
        )

    output_format = _extension(output_file)
    if output_format is None or output_format not in OUTPUT_FORMATS:
        raise CliError(
            f"Output file extension '.{output_format}' of {output_file} is not a supported output format.",
            code="unsupported_output_format",
            operation=operation,
            parameter="output",
            valid_values=sorted(OUTPUT_FORMATS),
            hint="Use an extension such as .json, .mgf, .msp or .pickle.",
        )

    if output_format == "pickle" and args.export_style != "matchms":
        raise CliError(
            f"Pickle export only supports export style 'matchms', got '{args.export_style}'.",
            code="invalid_parameter",
            operation=operation,
            parameter="export_style",
            valid_values=["matchms"],
        )

    if args.append and output_format not in ("mgf", "msp"):
        raise CliError(
            f"Appending is not supported for '.{output_format}' output files.",
            code="invalid_parameter",
            operation=operation,
            parameter="append",
            valid_values=["mgf", "msp"],
            hint="Only .mgf and .msp output files support --append.",
        )

    if os.path.exists(output_file) and not args.append:
        raise CliError(
            f"The specified output file: {output_file} already exists.",
            code="file_exists",
            operation=operation,
            input_file=output_file,
            hint="Choose a different output file or pass --append to append to it.",
        )

    base_name, descriptions = _build_pipeline(args, operation)
    processor = _make_processor(descriptions, operation)

    try:
        collection = load_ms2_dataset(input_file, ftype=args.ftype)
    except TypeError as exc:
        raise CliError(
            f"The file extension of {input_file} is not recognized as a supported input format.",
            code="unsupported_input_format",
            operation=operation,
            input_file=input_file,
            valid_values=sorted(INPUT_FORMATS),
            hint="Use an extension such as .mgf, .msp, .mzml, .mzxml, .json or .pickle, "
            "or select the file type explicitly with --ftype.",
        ) from exc
    except ValueError as exc:
        if "at least one Spectrum" in str(exc):
            raise CliError(
                f"No spectra were loaded from {input_file}; nothing to filter.",
                code="empty_spectra",
                operation=operation,
                input_file=input_file,
            ) from exc
        raise

    n_in = len(collection)
    if n_in == 0:
        raise CliError(
            f"No spectra were loaded from {input_file}; nothing to filter.",
            code="empty_spectra",
            operation=operation,
            input_file=input_file,
        )

    report = processor.create_processing_report() if args.report else None
    processed = processor.process_collection(collection, processing_report=report)
    if processed is None:
        n_out = 0
    else:
        n_out = len(processed)

    save_spectra(
        list(processed) if processed is not None else [],
        output_file,
        export_style=args.export_style,
        append=args.append,
    )

    payload = {
        "ok": True,
        "operation": CLI_COMMAND,
        "input_file": input_file,
        "input_format": input_format,
        "output_file": output_file,
        "output_format": output_format,
        "pipeline": base_name,
        "n_spectra_in": n_in,
        "n_spectra_out": n_out,
        "n_removed": n_in - n_out,
        "filters": [{"name": name, **({"parameters": params} if params else {})} for name, params in descriptions],
    }

    if report is not None:
        rows = _report_rows(report)
        report_file = _report_path(output_file)
        report_payload = {
            "operation": CLI_COMMAND,
            "pipeline": base_name,
            "input_file": input_file,
            "output_file": output_file,
            "n_spectra_in": n_in,
            "n_spectra_out": n_out,
            "n_removed": n_in - n_out,
            "steps": rows,
        }
        with open(report_file, "w", encoding="utf-8") as fh:
            json.dump(report_payload, fh, indent=2, default=str)
            fh.write("\n")
        payload["report_file"] = report_file
        payload["report"] = {"steps": rows}

    if ctx.machine_mode:
        ctx.write_json(payload)
    else:
        ctx.write_text(_format_human(payload))
    return 0


def _format_human(payload: dict) -> str:
    lines = [
        f"Filter run: {payload['input_file']} -> {payload['output_file']}",
        f"  pipeline:   {payload['pipeline']}",
        f"  spectra in: {payload['n_spectra_in']}",
        f"  spectra out: {payload['n_spectra_out']}",
        f"  removed:    {payload['n_removed']}",
        "",
        "Filters (in execution order):",
        human_table(
            ["name", "parameters"],
            [[f["name"], f.get("parameters")] for f in payload["filters"]],
        ),
    ]

    if payload.get("report_file"):
        lines += [
            "",
            "Processing report (per filter):",
            human_table(
                [key for _, key in REPORT_COLUMNS],
                [[step.get(key) for _, key in REPORT_COLUMNS] for step in payload["report"]["steps"]],
            ),
            f"  saved to: {payload['report_file']}",
        ]

    lines.append("")
    lines.append("Run with --json for the machine-readable version.")
    return "\n".join(lines)
