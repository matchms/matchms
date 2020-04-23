from typing import Union
from matchms.utils import mol_converter
from matchms import Spectrum
from .has_valid_inchikey import has_valid_inchikey
from .has_valid_inchi import has_valid_inchi


def derive_inchikey_from_inchi(spectrum_in) -> Union[Spectrum, None]:
    """Find missing InchiKey and derive from Inchi where possible."""

    spectrum = spectrum_in.clone()

    if has_valid_inchi(spectrum) and not has_valid_inchikey(spectrum):
        inchi = spectrum.get("inchi")
        inchikey = mol_converter(inchi, "inchi", "inchikey")
        if not inchikey:
            print("Could not convert InChI", inchi, "to inchikey.")
            inchikey = 'n/a'
        spectrum.set("inchikey", inchikey)

    return spectrum
