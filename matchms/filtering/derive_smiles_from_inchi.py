from ..typing import SpectrumType
from ..utils import convert_inchi_to_smiles
from ..utils import is_valid_inchi
from ..utils import is_valid_smiles


def derive_smiles_from_inchi(spectrum_in: SpectrumType) -> SpectrumType:
    """Find missing smiles and derive from Inchi where possible."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    inchi = spectrum.get("inchi")
    smiles = spectrum.get("smiles")

    if not is_valid_smiles(smiles) and is_valid_inchi(inchi):
        smiles = convert_inchi_to_smiles(inchi)
        if smiles:
            smiles = smiles.rstrip()
            spectrum.set("smiles", smiles)
        else:
            print("Could not convert InChI", inchi, "to smiles.")

    return spectrum
