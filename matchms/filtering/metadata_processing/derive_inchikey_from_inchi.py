import logging
from ..metadata_utils import (convert_inchi_to_inchikey, is_valid_inchi,
                              is_valid_inchikey)
from ..typing import SpectrumType


logger = logging.getLogger("matchms")


def derive_inchikey_from_inchi(spectrum_in: SpectrumType) -> SpectrumType:
    """Find missing InchiKey and derive from Inchi where possible."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    inchi = spectrum.get("inchi")
    inchikey = spectrum.get("inchikey")

    if is_valid_inchi(inchi) and not is_valid_inchikey(inchikey):
        inchikey = convert_inchi_to_inchikey(inchi)
        if inchikey:
            spectrum.set("inchikey", inchikey)
            logger.info("Added InChIKey %s to metadata (was converted from inchi)", inchikey)
        else:
            logger.warning("Could not convert InChI %s to inchikey.", inchi)

    return spectrum
