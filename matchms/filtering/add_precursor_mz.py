import logging
from .metadata_filters import _add_precursor_mz_metadata


logger = logging.getLogger("matchms")


def add_precursor_mz(spectrum_in):
    """Add precursor_mz to correct field and make it a float.

    For missing precursor_mz field: check if there is "pepmass"" entry instead.
    For string parsed as precursor_mz: convert to float.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    metadata_updated = _add_precursor_mz_metadata(spectrum.metadata)
    spectrum.metadata = metadata_updated
    return spectrum
