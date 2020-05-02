from ..utils import mol_converter
from ..typing import SpectrumType
from ..metadata_entry_testing import entry_is_empty, is_valid_inchikey


def clean_inchis(spectrum_in: SpectrumType, rescue_smiles=True) -> SpectrumType:
    """Make inchi style more consistent and wrongly given smiles.

    Args:
    ----
    spectrum_in: matchms.Spectrum()
        Input spectrum.
    rescue_smiles: bool
        If True, check if smiles is accidentaly given in inchi field.
        Default is True.

    Read spectrum, look for inchi. Then:
    1) Make line with inchi homogeneously looking like: '"InChI=..."'
    2) Test if inchi is actually inchikey. If so, add as inchikey to metadata
    unless inchikey already exists.
    3) if rescue_smiles is True then try to detect inchi that are actually smiles
    and convert to proper inchi.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if entry_is_empty(spectrum, "inchi"):
        spectrum.set("inchi", "N/A")
        return spectrum

    inchi = spectrum.get("inchi").replace(" ", "")
    inchi_core = inchi.strip().split("InChI=")[-1].replace('"', '')

    # Test if inchi is actually inchikey
    if is_valid_inchikey(inchi_core):
        print("Inchikey", inchi_core, "found instead of InChI.")
        if not is_valid_inchikey(spectrum.get("inchikey")):
            spectrum.set("inchikey", inchi_core)

        spectrum.set("inchi", "N/A")
        return spectrum

    # Test if inchi is actually smiles
    if inchi_core[0] in ["C", "c", "O", "N", "n", "B"] and rescue_smiles:
        inchi = mol_converter(inchi_core, "smi", "inchi")
        if not inchi or len(inchi) < 12:
            spectrum.set("inchi", "N/A")
            return spectrum
        print("New inchi added:", inchi.replace("\n", ""))
        print("Derived inchi from assumed smile:", inchi_core)

    # Make inchi string style consistent
    inchi = inchi.strip().split("InChI=")[-1].replace('"', '').replace("\n", "")
    inchi = '"InChI=' + inchi + '"'
    spectrum.set("inchi", inchi)
    return spectrum
