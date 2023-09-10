import logging
from matchms import Spectrum
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import \
    get_monoisotopic_neutral_mass
from ..filter_utils.load_known_adducts import load_known_adducts
from ..metadata_processing.repair_parent_mass_is_mol_wt import repair_parent_mass_is_mol_wt
from matchms.typing import SpectrumType
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter


logger = logging.getLogger("matchms")


class RepairAdductBasedOnSmiles(BaseSpectrumFilter):
    def __init__(self, mass_tolerance, accept_parent_mass_is_mol_wt):
        self.mass_tolerance = mass_tolerance
        self.accept_parent_mass_is_mol_wt = accept_parent_mass_is_mol_wt

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        changed_spectrum = spectrum.clone()

        precursor_mz = changed_spectrum.get("precursor_mz")
        ion_mode = changed_spectrum.get("ionmode")
        if ion_mode not in ("positive", "negative"):
            logger.warning("Ionmode: %s not positive or negative, first run derive_ionmode",
                            ion_mode)
            return changed_spectrum
        if precursor_mz is None:
            logger.warning("Precursor_mz is None, first run add_precursor_mz")
            return changed_spectrum
    
        adducts_df = load_known_adducts()
        smiles_mass = get_monoisotopic_neutral_mass(changed_spectrum.get("smiles"))
        parent_masses = (precursor_mz - adducts_df["correction_mass"]) / adducts_df["mass_multiplier"]
        mass_differences = abs(parent_masses-smiles_mass)
    
        # Select the lowest value
        smalles_mass_index = mass_differences.idxmin()
        parent_mass = parent_masses[smalles_mass_index]
        adduct = adducts_df.iloc[smalles_mass_index]["adduct"]
        # Change spectrum. This spectrum will only be returned if the mass difference is smaller than mass tolerance
        changed_spectrum.set("parent_mass", parent_mass)
        changed_spectrum.set("adduct", adduct)
        if mass_differences[smalles_mass_index] < self.mass_tolerance:
            logger.info("Adduct was set from %s to %s",
                        spectrum.get('adduct'), adduct)
            return changed_spectrum
        if self.accept_parent_mass_is_mol_wt:
            changed_spectrum = repair_parent_mass_is_mol_wt(changed_spectrum, self.mass_tolerance)
            if abs(changed_spectrum.get("parent_mass") - smiles_mass) < self.mass_tolerance:
                logger.info("Adduct was set from %s to %s",
                            spectrum.get('adduct'), adduct)
                return changed_spectrum
        return spectrum
