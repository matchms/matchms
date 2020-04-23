from typing import Union
from matchms.utils import mol_converter
from matchms import Spectrum
from .has_valid_smiles import has_valid_smiles
from .has_valid_inchi import has_valid_inchi


def derive_smiles_from_inchi(spectrum_in) -> Union[Spectrum, None]:
    """Find missing smiles and derive from Inchi where possible."""

    spectrum = spectrum_in.clone()

    if has_valid_inchi(spectrum) and not has_valid_smiles(spectrum):
        inchi = spectrum.get("inchi")
        smiles = mol_converter(inchi, "inchi", "smi")
        if not smiles:
            print("Could not convert InChI", inchi, "to smiles.")
            smiles = 'n/a'
        smiles = smiles.replace('\n', '').replace('\t', '').replace('\r', '')
        spectrum.set("smiles", smiles)

    return spectrum
