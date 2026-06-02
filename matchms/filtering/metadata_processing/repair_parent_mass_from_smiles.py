import logging
from matchms import SpectraCollection, Spectrum
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import (
    get_monoisotopic_neutral_mass,
)
from matchms.filtering.filter_utils.metadata_conversions import (
    apply_metadata_row_filter,
    apply_metadata_updates_to_spectrum,
    as_float_or_none,
    as_string_or_none,
)
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _repair_parent_mass_from_smiles(
    metadata,
    mass_tolerance: float = 0.1,
) -> dict:
    """Return metadata updates that make parent_mass match the smiles mass."""
    smiles = as_string_or_none(metadata.get("smiles"))
    smiles_mass = get_monoisotopic_neutral_mass(smiles)

    if smiles_mass is None:
        return {}

    parent_mass = as_float_or_none(metadata.get("parent_mass"))

    if parent_mass is None:
        return {"parent_mass": smiles_mass}

    if abs(parent_mass - smiles_mass) > mass_tolerance:
        return {"parent_mass": smiles_mass}

    return {}


def _repair_parent_mass_from_smiles_spectrum(
    spectrum_in: Spectrum,
    mass_tolerance: float = 0.1,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Set parent mass to match smiles mass if not already close."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    updates = _repair_parent_mass_from_smiles(
        spectrum.metadata,
        mass_tolerance=mass_tolerance,
    )

    return apply_metadata_updates_to_spectrum(spectrum, updates)


def _repair_parent_mass_from_smiles_collection(
    spectrum_in: SpectraCollection,
    mass_tolerance: float = 0.1,
    clone: bool | None = True,
) -> SpectraCollection:
    """Set parent mass to match smiles mass where possible for a SpectraCollection."""
    target = spectrum_in.copy() if clone else spectrum_in

    target.apply_to_metadata_rows(
        apply_metadata_row_filter,
        row_filter=_repair_parent_mass_from_smiles,
        mass_tolerance=mass_tolerance,
        inplace=True,
    )

    return target


repair_parent_mass_from_smiles = collection_filter(
    _repair_parent_mass_from_smiles_spectrum,
    collection_impl=_repair_parent_mass_from_smiles_collection,
)
