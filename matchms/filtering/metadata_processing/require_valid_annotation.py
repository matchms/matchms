from matchms import Spectrum
from matchms.filtering.filters.require_valid_annotation import RequireValidAnnotation


def require_valid_annotation(spectrum: Spectrum):
    """Removes spectra that are not fully annotated (correct and matching, smiles, inchi and inchikey)"""

    spectrum = RequireValidAnnotation().process(spectrum)
    return spectrum
