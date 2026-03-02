from logging import Logger
from matchms.filtering.metadata_processing.add_precursor_mz import _convert_precursor_mz


def get_valid_precursor_mz(spectrum, logger: Logger) -> float:
    """Extract valid precursor_mz from spectrum if possible. If not raise exception."""
    message_precursor_missing = "Precursor_mz missing. Apply 'add_precursor_mz' filter first."
    message_precursor_no_number = "Precursor_mz must be int or float. Apply 'add_precursor_mz' filter first."
    message_precursor_below_0 = "Expect precursor to be positive number. Apply 'require_precursor_mz' first."

    precursor_mz = spectrum.get("precursor_mz", None)
    if not isinstance(precursor_mz, (int, float)):
        logger.warning(message_precursor_no_number)
    precursor_mz = _convert_precursor_mz(precursor_mz)
    assert precursor_mz is not None, message_precursor_missing
    assert precursor_mz > 0, message_precursor_below_0
    return precursor_mz
