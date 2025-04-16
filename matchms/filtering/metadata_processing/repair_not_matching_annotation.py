import logging
from typing import Optional
from matchms import Spectrum
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    convert_inchi_to_inchikey,
    convert_inchi_to_smiles,
    convert_smiles_to_inchi,
    is_valid_inchi,
    is_valid_inchikey,
    is_valid_smiles,
)
from matchms.filtering.metadata_processing.require_parent_mass_match_smiles import _check_smiles_and_parent_mass_match
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def repair_not_matching_annotation(spectrum_in: Spectrum, clone: Optional[bool] = True) -> Optional[SpectrumType]:
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
    clone:
        Optionally clone the Spectrum.

    Returns:
    -------
    Spectrum
        A cloned version of the input spectrum with corrected annotations. If the input
        spectrum is `None`, it returns `None`.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    if not _check_repairing_is_possible(spectrum.get("smiles"), spectrum.get("inchi"), spectrum.get("inchikey")):
        # Repairing is not possible, since not all annotations are valid entries.
        return spectrum

    inchikey_from_smiles = convert_inchi_to_inchikey(convert_smiles_to_inchi(spectrum.get("smiles")))
    inchikey_from_inchi = convert_inchi_to_inchikey(spectrum.get("inchi"))

    # Check if SMILES and InChI match
    if inchikey_from_smiles[:14] != inchikey_from_inchi[:14]:
        spectrum = _repair_smiles_inchi(spectrum)

    # Check if the InChIKey matches the InChI
    correct_inchikey = convert_inchi_to_inchikey(spectrum.get("inchi"))
    if correct_inchikey and (spectrum.get("inchikey")[:14] == correct_inchikey[:14]):
        return spectrum

    logger.info("The inchikey has been changed from %s to %s", spectrum.get("inchikey"), correct_inchikey)
    spectrum.set("inchikey", correct_inchikey)
    return spectrum


def _check_repairing_is_possible(smiles, inchi, inchikey) -> bool:
    """Returns True if smiles, inchi and inchikey all are valid.

    If False the appropriate logging message is given.
    """
    valid_smiles = is_valid_smiles(smiles)
    valid_inchi = is_valid_inchi(inchi)
    valid_inchikey = is_valid_inchikey(inchikey)
    if valid_smiles and valid_inchi and valid_inchikey:
        # All annotation is valid, however the different annotations, might not correspond to the same compound.
        # Therefore repair_not_matching_annotation can be run.
        return True
    if not valid_smiles and not valid_inchi and not valid_inchikey:
        # There is no annotation available
        logger.info("No valid annotation was available for the spectrum, so repair_not_matching_annotation was not run")
    elif valid_smiles or valid_inchi or valid_inchikey:
        # At least one of the annotations, but some are not.
        # Since if valid_smiles and valid_inchi and valid_inchikey was False
        logger.warning(
            "Please first run repair_inchi_from_smiles, repair_smiles_from_inchi and repair_inchikey. "
            "The spectrum had partly valid annotations, "
            "this shows that these repair functions were not yet run."
        )
    return False


def _repair_smiles_inchi(spectrum):
    """Repairs mismatching smiles and inchi. The smile or inchi that matches the parent mass is chosen"""
    smiles = spectrum.get("smiles")
    parent_mass = spectrum.get("parent_mass")
    inchi = spectrum.get("inchi")
    smiles_from_inchi = convert_inchi_to_smiles(inchi)

    smiles_correct = _check_smiles_and_parent_mass_match(smiles, parent_mass, 0.1)
    inchi_correct = _check_smiles_and_parent_mass_match(smiles_from_inchi, parent_mass, 0.1)

    if smiles_correct and inchi_correct:
        logger.warning(
            "The SMILES and InChI are not matching, but both match the parent mass. SMILES = %s, InChI = %s",
            smiles,
            inchi,
        )
    elif smiles_correct and not inchi_correct:
        inchi_from_smiles = convert_smiles_to_inchi(spectrum.get("smiles"))
        # Repair by using inchi generated from SMILES
        logger.info((
                "The InChI has been changed from %s to %s. The new InChI matches the parent mass, while the old one "
                "did not"
            ),
            inchi,
            inchi_from_smiles,
        )
        spectrum.set("inchi", inchi_from_smiles)
    elif inchi_correct and not smiles_correct:
        # Repair by using SMILES generated from the inchi
        logger.info((
            "The SMILES has been changed from %s to %s to match the InChI. The new SMILES matches the parent mass, "
            "while the old one did not",
            ),
            smiles,
            smiles_from_inchi,
        )
        spectrum.set("smiles", smiles_from_inchi)
    else:
        logger.warning(
            "Both the Smiles %s and the inchi %s do not match the parent mass %f", smiles, inchi, parent_mass
        )
    return spectrum
