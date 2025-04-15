import logging
from typing import Optional
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    convert_smiles_to_inchi,
    is_valid_inchi,
    is_valid_smiles,
)
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def derive_inchi_from_smiles(spectrum_in: SpectrumType, clone: Optional[bool] = True) -> Optional[SpectrumType]:
    """Find missing Inchi and derive from smiles where possible.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with added INCHI, or `None` if not present.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in
    inchi = spectrum.get("inchi")
    smiles = spectrum.get("smiles")

    if not is_valid_inchi(inchi) and is_valid_smiles(smiles):
        inchi = convert_smiles_to_inchi(smiles)
        if inchi:
            inchi = inchi.rstrip()
            spectrum.set("inchi", inchi)
            logger.info("Added InChI (%s) to metadata (was converted from smiles).", inchi)
        else:
            logger.warning("Could not convert smiles %s to InChI.", smiles)

    return spectrum
