import logging
import numpy as np
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.metadata_conversions import apply_metadata_row_filter
from matchms.filtering.metadata_processing.repair_adduct_and_parent_mass_based_on_smiles import (
    _repair_adduct_and_parent_mass_based_on_smiles,
    repair_adduct_and_parent_mass_based_on_smiles,
)
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType
from .repair_parent_mass_is_molar_mass import (
    _repair_parent_mass_is_molar_mass,
    repair_parent_mass_is_molar_mass,
)
from .repair_smiles_of_salts import (
    _repair_smiles_of_salts,
    repair_smiles_of_salts,
)
from .require_parent_mass_match_smiles import _check_smiles_and_parent_mass_match


logger = logging.getLogger("matchms")


def _repair_parent_mass_match_smiles_wrapper_spectrum(
    spectrum_in: SpectrumType,
    mass_tolerance: float = 0.2,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Repair a mismatch between parent mass and smiles mass.

    The filter tries several increasingly involved repair steps:
    first salt removal from SMILES, then correction of molar mass to
    monoisotopic mass, then adduct/parent mass repair based on SMILES.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    filters_to_apply = [
        repair_smiles_of_salts,
        repair_parent_mass_is_molar_mass,
        repair_adduct_and_parent_mass_based_on_smiles,
    ]

    for filter_function in filters_to_apply:
        if _check_smiles_and_parent_mass_match(
            smiles=spectrum.get("smiles"),
            parent_mass=spectrum.get("parent_mass"),
            mass_tolerance=mass_tolerance,
        ):
            return spectrum

        spectrum = filter_function(
            spectrum,
            mass_tolerance=mass_tolerance,
            clone=False,
        )

    return spectrum


def _repair_parent_mass_match_smiles_wrapper_collection(
    spectrum_in: SpectraCollection,
    mass_tolerance: float = 0.2,
    clone: bool | None = True,
) -> SpectraCollection:
    """Repair parent_mass/smiles mismatches in a SpectraCollection."""
    target = spectrum_in.copy() if clone else spectrum_in

    metadata_filters_to_apply = [
        _repair_smiles_of_salts,
        _repair_parent_mass_is_molar_mass,
        _repair_adduct_and_parent_mass_based_on_smiles,
    ]

    for metadata_filter in metadata_filters_to_apply:
        needs_repair = ~_parent_mass_matches_smiles_mask(
            target,
            mass_tolerance=mass_tolerance,
        )

        if not needs_repair.any():
            return target

        target.apply_to_metadata_rows(
            apply_metadata_row_filter,
            row_mask=needs_repair,
            row_filter=metadata_filter,
            mass_tolerance=mass_tolerance,
            inplace=True,
        )

    return target


def _parent_mass_matches_smiles_mask(
    collection: SpectraCollection,
    mass_tolerance: float,
) -> np.ndarray:
    """Return True for rows where smiles and parent_mass already match."""
    metadata = collection.metadata

    if "smiles" not in metadata.columns or "parent_mass" not in metadata.columns:
        return np.zeros(len(collection), dtype=bool)

    return np.array(
        [
            _check_smiles_and_parent_mass_match(
                smiles=row.get("smiles"),
                parent_mass=row.get("parent_mass"),
                mass_tolerance=mass_tolerance,
            )
            for _, row in metadata.iterrows()
        ],
        dtype=bool,
    )


repair_parent_mass_match_smiles_wrapper = collection_filter(
    _repair_parent_mass_match_smiles_wrapper_spectrum,
    collection_impl=_repair_parent_mass_match_smiles_wrapper_collection,
)