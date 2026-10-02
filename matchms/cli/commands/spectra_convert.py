"""CLI command: matchms spectra convert.

Loads a spectra file via :func:`~matchms.importing.load_ms2_dataset` and writes
it back out with :func:`~matchms.exporting.save_spectra` (e.g. ``in.mgf`` to
``out.msp``). The input type is resolved from the input file extension against
the input formats known to :mod:`matchms.importing`; the output type is
resolved from the output file extension against the output formats known to
:mod:`matchms.exporting`.
"""

import os
from matchms.cli.errors import CliError
from matchms.exporting import save_spectra
from matchms.exporting.save_spectra import SUPPORTED_FILE_FORMATS as OUTPUT_FORMATS
from matchms.importing import load_ms2_dataset
from matchms.importing.load_spectra import SUPPORTED_FILE_FORMATS as INPUT_FORMATS


CLI_COMMAND = "spectra convert"

EXPORT_STYLES = ("matchms", "massbank", "nist", "riken", "gnps")


def run(args, ctx) -> int:
    """Run the `spectra convert` command."""
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

    n_spectra = len(collection)
    if n_spectra == 0:
        raise CliError(
            f"No spectra were loaded from {input_file}; nothing to convert.",
            code="empty_spectra",
            operation=operation,
            input_file=input_file,
        )

    save_spectra(list(collection), output_file, export_style=args.export_style, append=args.append)

    payload = {
        "ok": True,
        "operation": CLI_COMMAND,
        "input_file": input_file,
        "input_format": input_format,
        "output_file": output_file,
        "output_format": output_format,
        "n_spectra": n_spectra,
    }

    if ctx.machine_mode:
        ctx.write_json(payload)
    else:
        ctx.write_text(
            "\n".join(
                [
                    f"Converted {input_file} ({input_format}) -> {output_file} ({output_format}).",
                    f"  spectra: {n_spectra}",
                    f"  output:  {output_file}",
                ]
            )
        )
    return 0


def _extension(path: str) -> str | None:
    return os.path.splitext(path)[1].lower().lstrip(".") or None
