import itertools
import logging
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import \
    get_monoisotopic_neutral_mass
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


class RepairSmilesOfSalts(BaseSpectrumFilter):
    def __init__(self, mass_tolerance):
        self.mass_tolerance = mass_tolerance

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        smiles = spectrum.get("smiles")
        parent_mass = spectrum.get("parent_mass")
        possible_ion_combinations = RepairSmilesOfSalts._create_possible_ions(smiles)
        if not possible_ion_combinations:
            # It is not a salt
            return spectrum
        for ion, not_used_ions in possible_ion_combinations:
            ion_mass = get_monoisotopic_neutral_mass(ion)
            mass_diff = abs(parent_mass - ion_mass)
            # Check for Repair parent mass is mol wt did only return 1 spectrum. So not added as option for simplicity.
            if mass_diff < self.mass_tolerance:
                spectrum_with_ions = spectrum.clone()
                spectrum_with_ions.set("smiles", ion)
                spectrum_with_ions.set("salt_ions", not_used_ions)
                logger.info("Removed salt ions: %s from %s to match parent mass",
                            not_used_ions, smiles)
                return spectrum_with_ions
        logger.warning("None of the parts of the smile %s match the parent mass: %s",
                       smiles, parent_mass)
        return spectrum


    def _create_possible_ions(smiles):
        """Selects all possible ion combinations of a salt"""

        results = []
        if "." in smiles:
            single_ions = smiles.split(".")
            for r in range(1, len(single_ions) + 1):
                combinations = itertools.combinations(single_ions, r)
                for combination in combinations:
                    combined_ion = ".".join(combination)
                    removed_ions = single_ions.copy()
                    for used_ion in combination:
                        removed_ions.remove(used_ion)
                    removed_ions = ".".join(removed_ions)
                    results.append((combined_ion, removed_ions))
        return results
