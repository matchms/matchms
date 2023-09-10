import logging
import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.metadata_utils import (is_valid_inchi, is_valid_inchikey,
                                    is_valid_smiles)
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


class RepairSmilesFromCompoundName(BaseSpectrumFilter):
    def __init__(self, annotated_compound_names_file, mass_tolerance=0.1):
        self.annotated_compound_names_file = annotated_compound_names_file
        self.mass_tolerance = mass_tolerance

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        annotated_compound_names = RepairSmilesFromCompoundName._load_compound_name_annotations(self.annotated_compound_names_file)

        if RepairSmilesFromCompoundName._check_fully_annotated(spectrum):
            return spectrum
        compound_name = spectrum.get("compound_name")
        parent_mass = spectrum.get('parent_mass')

        if RepairSmilesFromCompoundName._is_plausible_name(compound_name) and parent_mass is not None:
            matching_compound_name = annotated_compound_names[annotated_compound_names["compound_name"] == compound_name]
            mass_differences = np.abs(matching_compound_name["monoisotopic_mass"]-parent_mass)
            within_mass_tolerance = matching_compound_name[mass_differences < self.mass_tolerance]
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
