import logging
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.derive_precursor_mz_and_parent_mass import (
    derive_parent_mass_from_metadata,
)
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import (
    get_monoisotopic_neutral_mass,
)
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none
from ...utils import get_first_common_element


logger = logging.getLogger("matchms")


_default_key = "parent_mass"
_accepted_keys = ["parentmass", "exact_mass"]
_accepted_types = (float, str, int)
_accepted_missing_entries = ["", "N/A", "NA", "n/a"]


def _add_parent_mass(
    metadata,
    estimate_from_adduct: bool = True,
    overwrite_existing_entry: bool = False,
    estimate_from_charge: bool = True,
) -> dict:
    """Add estimated parent mass to metadata if not present yet.

    Method to calculate the parent mass from given precursor m/z together with
    charge and/or adduct. Will take precursor m/z from ``precursor_mz`` as
    provided by running ``add_precursor_mz``.

    For ``estimate_from_adduct=True`` this function estimates the parent mass
    based on the mass and charge of known adducts. The table of known adduct
    properties can be found in ``matchms/data/known_adducts_table.csv``.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
    estimate_from_adduct
        When set to ``True``, use adduct to estimate actual molecular mass
        (``parent_mass``). Switches back to charge-based estimate if adduct does
        not match a known adduct. Default is ``True``.
    overwrite_existing_entry
        If ``False``, an existing parent-mass entry is kept. If ``True``, a newly
        computed value will replace existing ones. Default is ``False``.
    estimate_from_charge
        If ``True``, charge will be used to estimate the parent mass when adduct
        information is insufficient. Adducts of the form ``[M+H]+``,
        ``[M+H]2+``, ``[M-H]-`` etc. are assumed. Default is ``True``.
    clone
        Optionally clone the input before applying the filter. If ``False``, the
        input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Input object with added or updated ``parent_mass`` metadata, or ``None``
        if the input was ``None``.
    """
    parent_mass = _get_parent_mass(metadata)

    if parent_mass is not None and not overwrite_existing_entry:
        # Keep old behavior: normalize accepted aliases such as "parentmass" or
        # "exact_mass" into the default "parent_mass" key.
        return {"parent_mass": float(parent_mass)}

    parent_mass = derive_parent_mass_from_metadata(
        metadata,
        estimate_from_adduct=estimate_from_adduct,
        estimate_from_charge=estimate_from_charge,
    )

    if parent_mass is None:
        smiles = as_string_or_none(metadata.get("smiles"))
        parent_mass = get_monoisotopic_neutral_mass(smiles)

    if parent_mass is None:
        logger.warning("Not sufficient spectrum metadata to derive parent mass.")
        return {}

    return {"parent_mass": float(parent_mass)}


def _get_parent_mass(metadata):
    parent_mass_key = get_first_common_element(
        [_default_key] + _accepted_keys,
        metadata.keys(),
    )
    parent_mass = metadata.get(parent_mass_key)
    return _convert_entry_to_num(parent_mass)


def _convert_entry_to_num(entry):
    """Convert parent_mass to number if possible. Otherwise return None."""
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


add_parent_mass = metadata_update_filter(_add_parent_mass)