"""Shared file validation and loading helpers for the matchms CLI.

Every command that reads or writes spectra files needs the same handful of
checks. They live here once, with a single, consistent error vocabulary (see
:mod:`matchms.cli.constants`):

- a missing input file is always ``input_not_found``;
- an unsupported extension (input *or* output) is always ``unsupported_format``;
- an output directory that is missing or not writable is always ``save_failed``.

This replaces the per-command copies of these checks that had drifted apart.
"""

import os
from matchms.cli.constants import (
    ERROR_EMPTY_SPECTRA,
    ERROR_INPUT_NOT_FOUND,
    ERROR_SAVE_FAILED,
    ERROR_UNSUPPORTED_FORMAT,
    INPUT_FORMATS,
)
from matchms.cli.errors import CliError
from matchms.importing import load_ms2_dataset


INPUT_HINT = (
    "Use a supported extension such as .mgf, .msp, .mzml, .mzxml, .json or .pickle."
)


def extension_of(path: str) -> str | None:
    """The lowercase extension of *path* without the dot (``None`` if absent)."""
    return os.path.splitext(path)[1].lower().lstrip(".") or None


def validate_input_file(
    path: str,
    operation: str,
    *,
    kind: str = "input file",
    valid_values: list | None = None,
    hint: str | None = None,
) -> str:
    """Check *path* exists and has a supported input extension.

    Returns the detected input format (the extension without the dot) so the
    caller can pass it to :func:`load_collection`. Raises ``input_not_found``
    when the file is missing and ``unsupported_format`` when the extension is
    not in :data:`INPUT_FORMATS`.
    """
    if not os.path.exists(path):
        raise CliError(
            f"The specified {kind}: {path} does not exist.",
            code=ERROR_INPUT_NOT_FOUND,
            operation=operation,
            input_file=path,
            valid_values=valid_values if valid_values is not None else sorted(INPUT_FORMATS),
            hint=hint or "Expected a spectra file with a supported extension.",
        )
    file_format = extension_of(path)
    if file_format is None or file_format not in INPUT_FORMATS:
        raise CliError(
            f"{_kind(kind)} extension '.{file_format}' of {path} is not a supported input format. "
            "The input format is detected from the file extension only, so files with a "
            "non-standard extension cannot be loaded.",
            code=ERROR_UNSUPPORTED_FORMAT,
            operation=operation,
            input_file=path,
            valid_values=sorted(INPUT_FORMATS),
            hint=hint or INPUT_HINT,
        )
    return file_format


def validate_output_format(
    path: str,
    operation: str,
    *,
    valid_formats: list,
    parameter: str = "output",
    hint: str | None = None,
) -> str:
    """Check that *path* has one of *valid_formats* as its extension.

    Returns the detected output format. Raises ``unsupported_format`` otherwise.
    """
    output_format = extension_of(path)
    if output_format is None or output_format not in valid_formats:
        raise CliError(
            f"Output file extension '.{output_format}' of {path} is not a supported output format.",
            code=ERROR_UNSUPPORTED_FORMAT,
            operation=operation,
            parameter=parameter,
            valid_values=sorted(valid_formats),
            hint=hint,
        )
    return output_format


def validate_output_dir(output: str, operation: str) -> None:
    """Raise ``save_failed`` if the directory of *output* is missing or unwritable."""
    out_dir = os.path.dirname(os.path.abspath(output))
    if not os.path.isdir(out_dir):
        raise CliError(
            f"The output directory '{out_dir}' of '{output}' does not exist.",
            code=ERROR_SAVE_FAILED,
            operation=operation,
            input_file=output,
            hint="Create the directory first; the output directory must exist.",
        )
    if not os.access(out_dir, os.R_OK | os.W_OK):
        raise CliError(
            f"The output directory '{out_dir}' of '{output}' is not writable.",
            code=ERROR_SAVE_FAILED,
            operation=operation,
            input_file=output,
            hint="Choose an output directory that exists and is writable.",
        )


def load_collection(path: str, ftype: str, operation: str):
    """Load a spectra file into a :class:`~matchms.SpectraCollection`.

    A file that loads zero spectra is reported as ``empty_spectra``; any other
    load failure (bad format, parse error, ...) is left to surface as an
    unexpected error rather than being silently mislabelled.
    """
    try:
        return load_ms2_dataset(path, ftype=ftype)
    except ValueError as exc:
        if "at least one Spectrum" in str(exc):
            raise CliError(
                f"No spectra were loaded from {path}; nothing to process.",
                code=ERROR_EMPTY_SPECTRA,
                operation=operation,
                input_file=path,
            ) from exc
        raise


def _kind(kind: str) -> str:
    """A capitalised label for error messages from a *kind* such as 'query file'."""
    return kind[0].upper() + kind[1:]


__all__ = [
    "extension_of",
    "load_collection",
    "validate_input_file",
    "validate_output_dir",
    "validate_output_format",
]
