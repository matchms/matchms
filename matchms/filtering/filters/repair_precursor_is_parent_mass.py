import logging
from matchms import Spectrum
from matchms.filtering.filter_utils.derive_precursor_mz_and_parent_mass import \
    derive_precursor_mz_from_parent_mass
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import \
    get_monoisotopic_neutral_mass
from ..metadata_processing.require_parent_mass_match_smiles import require_parent_mass_match_smiles
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


class RepairPrecursorIsParentMass(BaseSpectrumFilter):
    def __init__(self, mass_tolerance):
        self.mass_tolerance = mass_tolerance

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        # Check if parent mass already matches smiles
        if require_parent_mass_match_smiles(spectrum, self.mass_tolerance) is not None:
            return spectrum

        precursor_mz = spectrum.get("precursor_mz")
        # Check if the precursor_mz can be calculated from the parent mass. If not skip this function
        if abs(precursor_mz - derive_precursor_mz_from_parent_mass(spectrum)) > self.mass_tolerance:
            return spectrum

        smiles = spectrum.get("smiles")
        smiles_mass = get_monoisotopic_neutral_mass(smiles)
        mass_difference = precursor_mz - smiles_mass
        if abs(mass_difference) < self.mass_tolerance:
            logger.info("Parent mass was changed from %s to %s", spectrum.get("parent_mass"), smiles_mass)
            spectrum.set("parent_mass", smiles_mass)
            new_precursor_mz = derive_precursor_mz_from_parent_mass(spectrum)
            if new_precursor_mz is not None:
                logger.info("Parent mass was changed from %s to %s", spectrum.get("precursor_mz"), new_precursor_mz)
                spectrum.set("precursor_mz", new_precursor_mz)
        return spectrum
