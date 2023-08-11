import logging
import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.metadata_utils import (is_valid_inchi, is_valid_inchikey,
                                    is_valid_smiles)


logger = logging.getLogger("matchms")


def repair_smiles_from_compound_name(spectrum_in: Spectrum,
                                     annotated_compound_names_file,
                                     mass_tolerance=0.1):
    """Adds annotations (smiles, inchi, inchikey) based on compound name

    Based on a table of compound names and smiles matches (stored in a csv file) this function
    adds the new annotations to the input spectrums if the smiles seem consistent with the available
    spectrum metadata (e.g., parent mass).
    This function can be used with csv files that are returned by the pubchem_lookup.py
    from matchmextras.

    Parameters
    ----------
    spectrum_in:
        The input spectrum.
    annotated_compound_names_file: str
        A csv file containing the compound names and matching smiles, inchi, inchikey
        and monoisotopic_mass. This can be created using the the pubchem_lookup.py from matchmextras.
    mass_tolerance.
        Acceptable mass difference between query compound and pubchem result.
    """
    annotated_compound_names = _load_compound_name_annotations(annotated_compound_names_file)
    if spectrum_in is None:
        return None
    spectrum = spectrum_in.clone()
    if _check_fully_annotated(spectrum):
        return spectrum
    compound_name = spectrum.get("compound_name")
    parent_mass = spectrum.get('parent_mass')

    if _is_plausible_name(compound_name) and parent_mass is not None:
        matching_compound_name = annotated_compound_names[annotated_compound_names["compound_name"] == compound_name]
        mass_differences = np.abs(matching_compound_name["monoisotopic_mass"]-parent_mass)
        within_mass_tolerance = matching_compound_name[mass_differences < mass_tolerance]
        if within_mass_tolerance.shape[0] > 0:
            # Select the match with the most
            best_match = within_mass_tolerance.loc[within_mass_tolerance["monoisotopic_mass"].idxmin()]
            spectrum.set("smiles", best_match["smiles"])
            spectrum.set("inchi", best_match["inchi"])
            spectrum.set("inchikey", best_match["inchikey"])
            logger.info("Added smiles %s based on the compound name %s", best_match["smiles"], compound_name)
            return spectrum
    return spectrum


def _load_compound_name_annotations(annotated_compound_names_file):
    """Loads in the annotated compound names and checks format"""
    annotated_compound_names = pd.read_csv(annotated_compound_names_file)
    assert list(annotated_compound_names.columns) == ["compound_name", "smiles", "inchi",
                                                      "inchikey", "monoisotopic_mass"], \
        "Choose a different annotated compound names file with columns compound_name, smiles, inchi, inchikey, monoisotopic_mass"
    return annotated_compound_names


def _check_fully_annotated(spectrum: Spectrum) -> bool:
    """Combine multiple check functions.
    Returns False if SMILES, InChIKey, or InChI are missing.
    """
    if not is_valid_smiles(spectrum.get("smiles")):
        return False
    if not is_valid_inchikey(spectrum.get("inchikey")):
        return False
    if not is_valid_inchi(spectrum.get("inchi")):
        return False
    return True


def _is_plausible_name(compound_name):
    """Simple check if it can be a compound name."""
    return isinstance(compound_name, str) and len(compound_name) > 4
