import logging
from matchms import Spectrum
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    convert_inchi_to_inchikey,
    convert_smiles_to_inchi,
    is_valid_inchi,
    is_valid_inchikey,
    is_valid_smiles,
)


logger = logging.getLogger("matchms")


def require_valid_annotation(spectrum: Spectrum):
    """Removes spectra that are not fully annotated (correct and matching, smiles, inchi and inchikey)"""
    if spectrum is None:
        return None
    smiles = spectrum.get("smiles")
    inchi = spectrum.get("inchi")
    inchikey = spectrum.get("inchikey")
    if not is_valid_smiles(smiles):
        logger.info("Removed spectrum since smiles is not valid. Incorrect smiles = %s", smiles)
        return None
    if not is_valid_inchikey(inchikey):
        logger.info("Removed spectrum since inchikey is not valid. Incorrect inchikey = %s", inchikey)
        return None
    if not is_valid_inchi(inchi):
        logger.info("Removed spectrum since inchi is not valid. Incorrect inchi = %s", inchi)
        return None
    if not _check_smiles_inchi_inchikey_match(smiles, inchi, inchikey):
        logger.info("Removed spectrum since smiles, inchi and inchikey do not match. Smiles = %s, inchi = %s, inchikey = %s", smiles, inchi, inchikey)
        return None
    return spectrum


def _check_smiles_inchi_inchikey_match(smiles, inchi, inchikey) -> bool:
    """Checks if smiles inchi and inchikey match"""
    # check if inchi matches the inchikey
    if not inchikey[:14] == convert_inchi_to_inchikey(inchi)[:14]:
        return False
    # check if smiles matches the inchikey (first convert to inchi followed by converting to inchikey)
    if not inchikey[:14] == convert_inchi_to_inchikey(convert_smiles_to_inchi(smiles))[:14]:
        return False
    return True
