import logging
from ..typing import SpectrumType
from ..utils import convert_inchi_to_smiles
from ..utils import is_valid_inchi
from ..utils import is_valid_smiles


logger = logging.getLogger("matchms")


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
            logger.info("Added smiles %s to metadata (was converted from InChI)", smiles)
        else:
            logger.warning("Could not convert InChI %s to smiles.", inchi)

    return spectrum
