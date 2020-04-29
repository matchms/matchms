from ..typing import SpectrumType
from ..utils import mol_converter
from ..metadata_entry_testing import entry_is_empty


def derive_inchi_from_smiles(spectrum_in: SpectrumType) -> SpectrumType:
    """Find missing Inchi and derive from smiles where possible."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if entry_is_empty(spectrum, "inchi") and not entry_is_empty(spectrum, "smiles"):
        smiles = spectrum.get("smiles")
        inchi = mol_converter(smiles, "smi", "inchi")
        if not inchi:
            # Second try: use smiley ("smy") parser
            inchi = mol_converter(smiles, "smy", "inchi")
        if not inchi:
            print("Could not convert smiles", smiles, "to InChI.")
            inchi = '"InChI=n/a"'
        inchi = inchi.replace('\n', '').replace('\t', '').replace('\r', '')
        spectrum.set("inchi", inchi)

    return spectrum
