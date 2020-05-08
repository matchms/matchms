from ..typing import SpectrumType
from ..utils import mol_converter, is_valid_inchi, is_valid_smiles


def derive_inchi_from_smiles(spectrum_in: SpectrumType) -> SpectrumType:
    """Find missing Inchi and derive from smiles where possible."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    inchi = spectrum.get("inchi")
    smiles = spectrum.get("smiles")

    if not is_valid_inchi(inchi) and is_valid_smiles(smiles):
        inchi = mol_converter(smiles, "smi", "inchi")
        if not inchi:
            # Second try: use smiley ("smy") parser
            inchi = mol_converter(smiles, "smy", "inchi")
        if inchi:
            inchi = inchi.replace('\n', '').replace('\t', '').replace('\r', '')
            spectrum.set("inchi", inchi)
        else:
            print("Could not convert smiles", smiles, "to InChI.")

    return spectrum
