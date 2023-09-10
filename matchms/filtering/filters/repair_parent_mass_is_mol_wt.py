import logging
from matchms import Spectrum
from matchms.filtering.filter_utils.derive_precursor_mz_and_parent_mass import \
    derive_precursor_mz_from_parent_mass
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import (
    get_molecular_weight_neutral_mass, get_monoisotopic_neutral_mass)
from ..metadata_processing.require_parent_mass_match_smiles import require_parent_mass_match_smiles
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


class RepairParentMassIsMolWt(BaseSpectrumFilter):
    def __init__(self, mass_tolerance: float):
        self.mass_tolerance = mass_tolerance

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        # Check if parent mass already matches smiles
        if require_parent_mass_match_smiles(spectrum, self.mass_tolerance) is not None:
            return spectrum
        # Check if the precursor_mz can be calculated from the parent mass. If not skip this function
        if abs(spectrum.get("precursor_mz") - derive_precursor_mz_from_parent_mass(spectrum)) > self.mass_tolerance:
            return spectrum
        # Check if parent mass matches the smiles mass
        parent_mass = spectrum.get("parent_mass")
        smiles = spectrum.get("smiles")
        smiles_mass = get_molecular_weight_neutral_mass(smiles)
        mass_difference = parent_mass - smiles_mass
        if abs(mass_difference) < self.mass_tolerance:
            correct_mass = get_monoisotopic_neutral_mass(smiles)
            spectrum.set("parent_mass", correct_mass)
            logger.info("Parent mass was mol_wt corrected from %s to %s",
                        parent_mass, correct_mass)
            precursor_mz = derive_precursor_mz_from_parent_mass(spectrum)
            logger.info("Precursor mz was derived from parent mass")
            spectrum.set("precursor_mz", precursor_mz)
        return spectrum
