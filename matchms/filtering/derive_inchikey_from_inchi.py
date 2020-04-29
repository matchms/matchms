from ..utils import mol_converter
from ..typing import SpectrumType
from ..metadata_entry_testing import entry_is_empty, is_valid_inchikey


def derive_inchikey_from_inchi(spectrum_in: SpectrumType) -> SpectrumType:
    """Find missing InchiKey and derive from Inchi where possible."""

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    inchikey = spectrum.get("inchikey")

    # TODO: replace with is_valid_inchi(inchi) if possible
    if not entry_is_empty(spectrum, "inchi") and not is_valid_inchikey(inchikey):
        inchi = spectrum.get("inchi")
        inchikey = mol_converter(inchi, "inchi", "inchikey")
        if not inchikey:
            print("Could not convert InChI", inchi, "to inchikey.")
            inchikey = 'n/a'
        spectrum.set("inchikey", inchikey)

    return spectrum
