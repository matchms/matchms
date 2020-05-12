from ..typing import SpectrumType
from ..utils import convert_inchi_to_inchikey
from ..utils import is_valid_inchi
from ..utils import is_valid_inchikey


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
        else:
            print("Could not convert InChI", inchi, "to inchikey.")

    return spectrum
