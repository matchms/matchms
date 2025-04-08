import logging
from typing import Optional
from matchms.filtering.filter_utils.derive_precursor_mz_and_parent_mass import derive_parent_mass_from_precursor_mz
from matchms.Spectrum import Spectrum
from ...utils import get_first_common_element
from ..filter_utils.get_neutral_mass_from_smiles import get_monoisotopic_neutral_mass


logger = logging.getLogger("matchms")


_default_key = "parent_mass"
_accepted_keys = ["parentmass", "exact_mass"]
_accepted_types = (float, str, int)
_accepted_missing_entries = ["", "N/A", "NA", "n/a"]


def add_parent_mass(
    spectrum_in: Spectrum, estimate_from_adduct: bool = True, overwrite_existing_entry: bool = False, estimate_from_charge: bool = True
) -> Optional[Spectrum]:
    """Add estimated parent mass to metadata (if not present yet).

    Method to calculate the parent mass from given precursor m/z together
    with charge and/or adduct. Will take precursor m/z from "precursor_mz"
    as provided by running `add_precursor_mz`.
    For estimate_from_adduct=True this function will estimate the parent mass based on
    the mass and charge of known adducts. The table of known adduct properties can be
    found under :download:`matchms/data/known_adducts_table.csv </../matchms/data/known_adducts_table.csv>`.

    Parameters
    ----------
    spectrum_in
        Input spectrum.
    estimate_from_adduct
        When set to True, use adduct to estimate actual molecular mass ("parent mass").
        Default is True. Switches back to charge-based estimate if adduct does not match
        a known adduct.
    overwrite_existing_entry
        Default is False. If set to True, a newly computed value will replace existing ones.
    estimate_from_charge
        Default is True. If set to True, the charge will be used to estimate the parent mass.
        Adduct of the form [M+H]+, [M+H]2+, [M-H]- etc are assumed.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    parent_mass = _get_parent_mass(spectrum.metadata)
    if parent_mass is not None and not overwrite_existing_entry:
        spectrum.set("parent_mass", parent_mass)
        return spectrum

    parent_mass = derive_parent_mass_from_precursor_mz(spectrum, estimate_from_adduct, estimate_from_charge=estimate_from_charge)

    if parent_mass is None:
        parent_mass = get_monoisotopic_neutral_mass(spectrum_in.get("smiles"))

    if parent_mass is None:
        logger.warning("Not sufficient spectrum metadata to derive parent mass.")
    else:
        spectrum.set("parent_mass", float(parent_mass))
    return spectrum


def _get_parent_mass(metadata):
    parent_mass_key = get_first_common_element([_default_key] + _accepted_keys, metadata.keys())
    parent_mass = metadata.get(parent_mass_key)
    parent_mass = _convert_entry_to_num(parent_mass)
    if parent_mass not in _accepted_missing_entries:
        return parent_mass
    return None


def _convert_entry_to_num(entry):
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
