import logging
from matchms import Spectrum
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    _check_fully_annotated, convert_inchi_to_inchikey, convert_inchi_to_smiles,
    convert_smiles_to_inchi, is_valid_inchi, is_valid_inchikey,
    is_valid_smiles)
from matchms.filtering.metadata_processing.require_parent_mass_match_smiles import \
    _check_smiles_and_parent_mass_match


logger = logging.getLogger("matchms")


def repair_not_matching_annotation(spectrum_in: Spectrum):
    """When smiles and inchi do not match the annotation that matches parent mass will be kept"""
    if spectrum_in is None:
        return None
    spectrum = spectrum_in.clone()
    # Check if smiles, inchi and inchikey are valid
    if _check_fully_annotated(spectrum):
        logger.warning("First run derive inchi_from_smiles, derive_inchikey_from_inchi and derive_smiles_from_inchi")
    smiles = spectrum.get("smiles")
    inchi = spectrum.get("inchi")
    inchikey = spectrum.get("inchikey")

    # Check if smiles and inchi match
    if convert_smiles_to_inchi(smiles) != inchi:
        # There is a mismatch between smiles and inchi
        parent_mass = spectrum.get("parent_mass")
        smiles_correct = _check_smiles_and_parent_mass_match(smiles, parent_mass, 0.1)
        inchi_correct = _check_smiles_and_parent_mass_match(convert_inchi_to_smiles(inchi), parent_mass, 0.1)
        if smiles_correct and inchi_correct:
            logger.warning("The smiles and inchi are not matching, but both match the parent mass. "
                           "Smiles = %s, inchi = %s, inchikey = %s", smiles, inchi, inchikey)
            return spectrum
        if smiles_correct and not inchi_correct:
            # repair the inchi from the smiles
            new_inchi = convert_smiles_to_inchi(smiles)
            spectrum.set("inchi", new_inchi)
            logger.info("The inchi has been changed from %s to %s"
                        "The new inchi matches the parent mass, while the old one did not", inchi, new_inchi)
        else:
            # repair the smiles from the inchi
            new_smiles = convert_inchi_to_smiles(inchi)
            spectrum.set("smiles", new_smiles)
            logger.info("The smiles has been changed from %s to %s, to match the inchi. "
                        "The new smiles matches the parent mass, while the old one did not", smiles, new_smiles)

    # Check if the inchikey matches the inchi
    new_inchikey = convert_inchi_to_inchikey(spectrum.get("inchi"))
    if is_valid_inchikey(inchikey):
        if inchikey == new_inchikey:
            # The inchi, smiles and inchikey all already matched
            return spectrum
    # The smiles and inchi (now) match, but the inchikey is still wrong
    # Repair inchikey
    logger.info("The inchikey has been changed from %s to %s", inchikey, new_inchikey)
    spectrum.set("inchikey", new_inchikey)
    return spectrum


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
        logger.info("Removed spectrum since smiles, inchi and inchikey do not match. "
                    "Smiles = %s, inchi = %s, inchikey = %s", smiles, inchi, inchikey)
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
