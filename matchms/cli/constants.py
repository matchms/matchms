"""Shared constants for the matchms CLI.

Single source of truth for values that several commands need: the supported
input/output file formats, the metadata export styles, and the error-code
vocabulary reported in every structured error payload.

The error codes are the CLI's machine-readable API: a consumer branches on the
``error`` field, so the vocabulary must stay stable and unambiguous. Each
failure condition has exactly one code (e.g. a missing file is always
``input_not_found``, a bad extension -- input or output -- is always
``unsupported_format``).
"""
from matchms.exporting.save_spectra import EXPORT_STYLES
from matchms.exporting.save_spectra import SUPPORTED_FILE_FORMATS as OUTPUT_FORMATS
from matchms.importing.load_spectra import SUPPORTED_FILE_FORMATS as INPUT_FORMATS


# --- error codes --------------------------------------------------------------
# A missing input file.
ERROR_INPUT_NOT_FOUND = "input_not_found"
# A file extension (input or output) that is not a supported format.
ERROR_UNSUPPORTED_FORMAT = "unsupported_format"
# An output file that already exists and ``--append`` was not given.
ERROR_FILE_EXISTS = "file_exists"
# A spectra file that loaded zero spectra.
ERROR_EMPTY_SPECTRA = "empty_spectra"
# A flag/parameter/value that is malformed or out of range for its target.
ERROR_INVALID_PARAMETER = "invalid_parameter"
# A parameter that a filter/similarity requires but was not supplied.
ERROR_MISSING_PARAMETER = "missing_parameter"
# Input data that is structurally invalid for the operation (e.g. a library that
# violates a peak-separation requirement), as opposed to a bad parameter value.
ERROR_INVALID_INPUT = "invalid_input"
# A generic failure while computing (loading, indexing, scoring, searching).
ERROR_COMPUTE_ERROR = "compute_error"
# A failure writing an output artifact.
ERROR_SAVE_FAILED = "save_failed"
# A required dependency of the selected method is not installed.
ERROR_MISSING_DEPENDENCY = "missing_dependency"
# A similarity that cannot do what the command asks (e.g. no index support, or
# ``--mode sparse`` on a dense-only method).
ERROR_UNSUPPORTED_METHOD = "unsupported_method"
# A requested name (method, filter, pipeline, id field, score field) is unknown.
ERROR_UNKNOWN_VALUE = "unknown_value"
# A dense matrix too large to materialize / to write in long form.
ERROR_MATRIX_TOO_LARGE = "matrix_too_large"
# A saved index that does not match the requested similarity configuration.
ERROR_INDEX_INCOMPATIBLE = "index_incompatible"


__all__ = [
    "ERROR_COMPUTE_ERROR",
    "ERROR_EMPTY_SPECTRA",
    "ERROR_FILE_EXISTS",
    "ERROR_INDEX_INCOMPATIBLE",
    "ERROR_INPUT_NOT_FOUND",
    "ERROR_INVALID_INPUT",
    "ERROR_INVALID_PARAMETER",
    "ERROR_MATRIX_TOO_LARGE",
    "ERROR_MISSING_DEPENDENCY",
    "ERROR_MISSING_PARAMETER",
    "ERROR_SAVE_FAILED",
    "ERROR_UNKNOWN_VALUE",
    "ERROR_UNSUPPORTED_FORMAT",
    "ERROR_UNSUPPORTED_METHOD",
    "EXPORT_STYLES",
    "INPUT_FORMATS",
    "OUTPUT_FORMATS",
]
