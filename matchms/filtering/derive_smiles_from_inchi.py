from ..typing import SpectrumType
from ..utils import is_valid_inchi
from ..utils import is_valid_smiles
from ..utils import mol_converter


def derive_smiles_from_inchi(spectrum_in: SpectrumType) -> SpectrumType:
    """Find missing smiles and derive from Inchi where possible."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    inchi = spectrum.get("inchi")
    smiles = spectrum.get("smiles")

    if not is_valid_smiles(smiles) and is_valid_inchi(inchi):
        smiles = mol_converter(inchi, "inchi", "smi")
        if smiles:
            smiles = smiles.replace('\n', '').replace('\t', '').replace('\r', '')
            spectrum.set("smiles", smiles)
        else:
            print("Could not convert InChI", inchi, "to smiles.")

    return spectrum
