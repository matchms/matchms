from ..utils import mol_converter, is_valid_inchi, is_valid_inchikey
from ..typing import SpectrumType


def derive_inchikey_from_inchi(spectrum_in: SpectrumType) -> SpectrumType:
    """Find missing InchiKey and derive from Inchi where possible."""

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    inchi = spectrum.get("inchi")
    inchikey = spectrum.get("inchikey")

    if is_valid_inchi(inchi) and not is_valid_inchikey(inchikey):
        inchikey = mol_converter(inchi, "inchi", "inchikey")
        if inchikey:
            spectrum.set("inchikey", inchikey)
        else:
            print("Could not convert InChI", inchi, "to inchikey.")

    return spectrum
