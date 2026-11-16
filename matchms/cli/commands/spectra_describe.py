"""CLI command: matchms spectra describe.

Loads a spectra file via :func:`~matchms.importing.load_ms2_dataset` and
reports the descriptive statistics of the collection, as computed by
:meth:`~matchms.spectra_collection.SpectraCollection.describe` (peak counts,
intensity sums and intensity entropy per spectrum, summarized over the
whole collection).
"""

import os
from matchms.cli.errors import CliError
from matchms.cli.output import human_table
from matchms.importing import load_ms2_dataset
from matchms.importing.load_spectra import SUPPORTED_FILE_FORMATS as INPUT_FORMATS


CLI_COMMAND = "spectra describe"

STATS_ROWS = ("count", "mean", "std", "min", "25%", "50%", "75%", "max")


def run(args, ctx) -> int:
    """Run the `spectra describe` command."""
    file = args.spectrumfile
    operation = f"{CLI_COMMAND} {file}"
    if not os.path.exists(file):
        raise CliError(
            f"The specified file: {file} does not exist.",
            code="file_not_found",
            operation=operation,
            input_file=file,
            valid_values=sorted(INPUT_FORMATS),
            hint="Expected a spectra file with a supported extension (e.g. .mgf, .msp, .mzml, .mzxml, .json, .pickle).",
        )

    collection = _load_collection(file, args.ftype, operation)
    stats = collection.describe()

    payload = {
        "ok": True,
        "operation": CLI_COMMAND,
        "input_file": file,
        "n_spectra": int(stats.attrs.get("num_spectra", len(collection))),
        "statistics": {
            metric: {str(stat_row): float(stats.loc[stat_row, metric]) for stat_row in STATS_ROWS}
            for metric in stats.columns
        },
    }

    if ctx.machine_mode:
        ctx.write_json(payload)
    else:
        ctx.write_text(_format_human(payload))
    return 0


def _load_collection(file: str, ftype: str | None, operation: str):
    try:
        return load_ms2_dataset(file, ftype=ftype)
    except TypeError as exc:
        raise CliError(
            f"The file extension of {file} is not recognized as a supported input format.",
            code="unsupported_input_format",
            operation=operation,
            input_file=file,
            valid_values=sorted(INPUT_FORMATS),
            hint="Use an extension such as .mgf, .msp, .mzml, .mzxml, .json or .pickle, "
            "or select the file type explicitly with --ftype.",
        ) from exc
    except AssertionError as exc:
        raise CliError(str(exc), code="file_not_found", operation=operation, input_file=file) from exc
    except ValueError as exc:
        # SpectraCollection rejects an empty input with a ValueError; other
        # ValueErrors (e.g. a metadata/fragment mismatch) are genuine bugs and
        # are re-raised so they surface as an internal_error.
        if "at least one Spectrum" in str(exc):
            raise CliError(
                f"No spectra were loaded from {file}; nothing to describe.",
                code="empty_spectra",
                operation=operation,
                input_file=file,
            ) from exc
        raise


def _format_human(payload: dict) -> str:
    lines = [
        f"SpectraCollection Describe: {payload['input_file']}",
        f"  spectra: {payload['n_spectra']}",
        "",
    ]
    columns = list(payload["statistics"])
    rows = [[row] + [payload["statistics"][column][row] for column in columns] for row in STATS_ROWS]
    lines.append(human_table(["stat"] + columns, rows))
    lines.append("")
    lines.append("Run with --json for the machine-readable version.")
    return "\n".join(lines)
