import logging
from matchms.filtering.filter_utils.derive_precursor_mz_and_parent_mass import \
    derive_parent_mass_from_precursor_mz
from matchms.typing import SpectrumType
from ...utils import get_first_common_element
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter


logger = logging.getLogger("matchms")


_default_key = "parent_mass"
_accepted_keys = ["parentmass", "exact_mass"]
_accepted_types = (float, str, int)
_accepted_missing_entries = ["", "N/A", "NA", "n/a"]


class AddParentMass(BaseSpectrumFilter):
    def __init__(self, estimate_from_adduct: bool = True, overwrite_existing_entry: bool = False):
        self.estimate_from_adduct = estimate_from_adduct
        self.overwrite_existing_entry = overwrite_existing_entry

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        parent_mass = self._get_parent_mass(spectrum.metadata)
        if parent_mass is not None and not self.overwrite_existing_entry:
            spectrum.set("parent_mass", parent_mass)
            return spectrum

        parent_mass = derive_parent_mass_from_precursor_mz(spectrum, self.estimate_from_adduct)

        if parent_mass is None:
            logger.warning("Not sufficient spectrum metadata to derive parent mass.")
        else:
            spectrum.set("parent_mass", float(parent_mass))
        return spectrum

    def _get_parent_mass(self, metadata):
        parent_mass_key = get_first_common_element([_default_key] + _accepted_keys,
                                                   metadata.keys())
        parent_mass = metadata.get(parent_mass_key)
        parent_mass = self._convert_entry_to_num(parent_mass)
        if parent_mass not in _accepted_missing_entries:
            return parent_mass
        return None

    def _convert_entry_to_num(self, entry):
        """Convert precursor_mz to number if possible. Otherwise return None."""
        if entry is None:
            return None
        if isinstance(entry, str) and entry in _accepted_missing_entries:
            return None
        if not isinstance(entry, _accepted_types):
            logger.warning("Found parent_mass of undefined type.")
            return None
        if isinstance(entry, str):
            try:
                return float(entry.strip())
            except ValueError:
                logger.warning("%s can't be converted to float.", entry)
                return None
        return entry
