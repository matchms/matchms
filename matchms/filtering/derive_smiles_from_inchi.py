from ..utils import mol_converter
from ..typing import SpectrumType
from ..metadata_entry_testing import entry_is_empty


def derive_smiles_from_inchi(spectrum_in: SpectrumType) -> SpectrumType:
    """Find missing smiles and derive from Inchi where possible."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if entry_is_empty(spectrum, "smiles") and not entry_is_empty(spectrum, "inchi"):
        inchi = spectrum.get("inchi")
        smiles = mol_converter(inchi, "inchi", "smi")
        if not smiles:
            print("Could not convert InChI", inchi, "to smiles.")
            smiles = 'n/a'
        smiles = smiles.replace('\n', '').replace('\t', '').replace('\r', '')
        spectrum.set("smiles", smiles)

    return spectrum
