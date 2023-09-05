import logging
from matchms import Spectrum
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    _check_fully_annotated, convert_inchi_to_inchikey, convert_inchi_to_smiles,
    convert_smiles_to_inchi, is_valid_inchi, is_valid_smiles)
from matchms.filtering.metadata_processing.require_parent_mass_match_smiles import \
    _check_smiles_and_parent_mass_match


logger = logging.getLogger("matchms")


def repair_not_matching_annotation(spectrum_in: Spectrum):
    """
    Repairs mismatches in a spectrum's annotations related to SMILES, InChI, and InChIKey.

    Given a spectrum, this function ensures that the provided SMILES, InChI, and InChIKey
    annotations are consistent with one another. If there are discrepancies, they are resolved
    as follows:

    1. If the SMILES and InChI do not match:
        - Both SMILES and InChI are checked against the parent mass.
        - The annotation that matches the parent mass is retained, and the other is regenerated.
    2. If the InChIKey does not match the InChI:
        - A new InChIKey is generated from the InChI and replaces the old one.

    Warnings and information logs are generated to track changes and potential issues.
    For correctness of InChIKey entries, only the first 14 characters are considered.

    Parameters:
    ----------
    spectrum_in : Spectrum
        The input spectrum containing annotations to be checked and repaired.

    Returns:
    -------
    Spectrum
        A cloned version of the input spectrum with corrected annotations. If the input
        spectrum is `None`, it returns `None`.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    smiles = spectrum.get("smiles")
    inchi = spectrum.get("inchi")
    inchikey = spectrum.get("inchikey")

    if not _check_fully_annotated(spectrum):
        if is_valid_inchi(inchi) or is_valid_smiles(smiles) or is_valid_smiles(inchikey):
            logger.warning("Please first run repair_inchi_from_smiles, repair_smiles_from_inchi and repair_inchikey. "
                           "The spectrum had partly valid annotations, "
                           "this shows that these repair functions were not yet run.")
        else:
            logger.info("No valid annotation was available for the spectrum, "
                        "so repair_not_matching_annotation was not run")
        return spectrum

    parent_mass = spectrum.get("parent_mass")
    inchi_from_smiles = convert_smiles_to_inchi(smiles)

    # Check if SMILES and InChI match
    if inchi_from_smiles != inchi:
        smiles_from_inchi = convert_inchi_to_smiles(inchi)
        spectrum = _repair_smiles_inchi(spectrum,
                                        smiles,
                                        inchi,
                                        smiles_from_inchi,
                                        inchi_from_smiles,
                                        parent_mass)

    # Check if the InChIKey matches the InChI
    correct_inchikey = convert_inchi_to_inchikey(spectrum.get("inchi"))
    if correct_inchikey and (inchikey[:14] == correct_inchikey[:14]):
        return spectrum

    logger.info("The inchikey has been changed from %s to %s", inchikey, correct_inchikey)
    spectrum.set("inchikey", correct_inchikey)
    return spectrum


def _repair_smiles_inchi(spectrum, smiles, inchi,
                         smiles_from_inchi, inchi_from_smiles,
                         parent_mass):
    # pylint: disable=too-many-arguments
    smiles_correct = _check_smiles_and_parent_mass_match(smiles, parent_mass, 0.1)
    inchi_correct = _check_smiles_and_parent_mass_match(smiles_from_inchi, parent_mass, 0.1)

    if smiles_correct and inchi_correct:
        logger.warning("The SMILES and InChI are not matching, but both match the parent mass. "
                       "SMILES = %s, InChI = %s", smiles, inchi)
    elif smiles_correct and not inchi_correct:
        # Repair by using inchi generated from SMILES
        logger.info("The InChI has been changed from %s to %s. "
                    "The new InChI matches the parent mass, while the old one did not", inchi, inchi_from_smiles)
        spectrum.set("inchi", inchi_from_smiles)
    else:
        # Repair by using SMILES generated from the inchi
        logger.info("The SMILES has been changed from %s to %s to match the InChI. "
                    "The new SMILES matches the parent mass, while the old one did not", smiles, smiles_from_inchi)
        spectrum.set("smiles", smiles_from_inchi)
    return spectrum
