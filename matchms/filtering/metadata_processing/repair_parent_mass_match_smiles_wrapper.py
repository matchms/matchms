import logging
import numpy as np
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.metadata_processing.repair_adduct_and_parent_mass_based_on_smiles import (
    repair_adduct_and_parent_mass_based_on_smiles,
    _repair_adduct_and_parent_mass_based_on_smiles_metadata,
)
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType
from .repair_parent_mass_is_molar_mass import (
    repair_parent_mass_is_molar_mass,
    _repair_parent_mass_is_molar_mass_metadata,
)
from .repair_smiles_of_salts import (
    repair_smiles_of_salts,
    _repair_smiles_of_salts_metadata,
)
from .require_parent_mass_match_smiles import _check_smiles_and_parent_mass_match


logger = logging.getLogger("matchms")


def _repair_parent_mass_match_smiles_wrapper_spectrum(
    spectrum_in: SpectrumType,
    mass_tolerance: float = 0.2,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Wrapper function for repairing a mismatch between parent mass and smiles mass

    Parameters:
    ----------
    spectrum_in : Spectrum
        The input spectrum containing annotations to be checked and repaired.
    mass_tolerance:
        Maximum allowed mass difference between the calculated parent mass and the neutral
        monoisotopic mass derived from the SMILES. Defaults to 0.2.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with repaired parent mass, or `None` if not present.
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
        _repair_smiles_of_salts_metadata,
        _repair_parent_mass_is_molar_mass_metadata,
        _repair_adduct_and_parent_mass_based_on_smiles_metadata,
    ]

    for metadata_filter in metadata_filters_to_apply:
        needs_repair = ~_parent_mass_matches_smiles_mask(
            target,
            mass_tolerance=mass_tolerance,
        )

        if not needs_repair.any():
            return target

        target = target.apply_to_metadata_rows(
            needs_repair,
            metadata_filter,
            mass_tolerance=mass_tolerance,
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
