from ..typing import SpectrumType
from ..utils import convert_smiles_to_inchi
from ..utils import is_valid_inchi
from ..utils import is_valid_smiles


def derive_inchi_from_smiles(spectrum_in: SpectrumType) -> SpectrumType:
    """Find missing Inchi and derive from smiles where possible."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    inchi = spectrum.get("inchi")
    smiles = spectrum.get("smiles")

    if not is_valid_inchi(inchi) and is_valid_smiles(smiles):
        inchi = convert_smiles_to_inchi(smiles)
        if inchi:
            inchi = inchi.rstrip()
            spectrum.set("inchi", inchi)
        else:
            print("Could not convert smiles", smiles, "to InChI.")

    return spectrum
