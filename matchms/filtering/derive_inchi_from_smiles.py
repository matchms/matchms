from matchms.utils import mol_converter
from .has_valid_inchi import has_valid_inchi
from .has_valid_smiles import has_valid_smiles


def derive_inchi_from_smiles(spectrum_in):
    """Find missing Inchi and derive from smiles where possible."""

    spectrum = spectrum_in.clone()

    if has_valid_smiles(spectrum) and not has_valid_inchi(spectrum):
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
