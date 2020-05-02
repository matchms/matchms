from ..metadata_entry_testing import is_valid_inchikey
from ..typing import SpectrumType


def clean_inchikeys(spectrum_in: SpectrumType) -> SpectrumType:
    """Harmonize metadata inchikey field and test if correct inchikey is given.

    Args:
    ----
    spectrum_in: matchms.Spectrum()
        Input spectrum.

    Read spectrum, look for 'inchikey'. If not found then check if "inchiaux"
    contains inchikey.
    """

    spectrum = spectrum_in.clone()
    inchikey = spectrum.get("inchikey")
    inchiaux = spectrum.get("inchiaux")

    # If no (correct) inchikey in metadata, but in "inchiaux"
    if not is_valid_inchikey(inchikey):
        if is_valid_inchikey(inchiaux):
            spectrum.set("inchikey", inchiaux)
        else:
            spectrum.set("inchikey", 'N/A')

    return spectrum
