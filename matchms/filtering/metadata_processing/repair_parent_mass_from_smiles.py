import logging
import pandas as pd
from matchms import Spectrum
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import (
    get_monoisotopic_neutral_mass,
)
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    as_string_or_none,
)
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _repair_parent_mass_from_smiles_value(smiles, parent_mass, mass_tolerance):
    """Return repaired parent mass if smiles mass can be used, otherwise original parent_mass."""
    smiles = as_string_or_none(smiles)
    smiles_mass = get_monoisotopic_neutral_mass(smiles)

    if smiles_mass is None:
        return parent_mass

    parent_mass_float = as_float_or_none(parent_mass)

    if parent_mass_float is None:
        return smiles_mass

    if abs(parent_mass_float - smiles_mass) > mass_tolerance:
        return smiles_mass

    return parent_mass


def _repair_parent_mass_from_smiles_metadata(
    metadata: pd.DataFrame,
    mass_tolerance: float = 0.1,
) -> pd.DataFrame:
    """Set parent_mass to smiles-derived monoisotopic mass where needed."""
    metadata = metadata.copy()

    if "smiles" not in metadata.columns:
        return metadata

    if "parent_mass" not in metadata.columns:
        metadata["parent_mass"] = None

    metadata["parent_mass"] = metadata["parent_mass"].astype("object")

    metadata["parent_mass"] = metadata.apply(
        lambda row: _repair_parent_mass_from_smiles_value(
            smiles=row.get("smiles"),
            parent_mass=row.get("parent_mass"),
            mass_tolerance=mass_tolerance,
        ),
        axis=1,
    )

    return metadata


def _repair_parent_mass_from_smiles_spectrum(
    spectrum_in: Spectrum,
    mass_tolerance: float = 0.1,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Set parent mass to match smiles mass if not already close.
    Parameters:
    ----------
    spectrum_in : Spectrum
        The input spectrum containing annotations to be checked and repaired.
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

    repaired_parent_mass = _repair_parent_mass_from_smiles_value(
        smiles=spectrum.get("smiles"),
        parent_mass=spectrum.get("parent_mass"),
        mass_tolerance=mass_tolerance,
    )

    spectrum.set("parent_mass", repaired_parent_mass)

    return spectrum


def _repair_parent_mass_from_smiles_collection(
    spectrum_in: SpectraCollection,
    mass_tolerance: float = 0.1,
    clone: bool | None = True,
) -> SpectraCollection:
    """Set parent mass to match smiles mass where possible for a SpectraCollection."""
    target = spectrum_in.copy() if clone else spectrum_in

    target = target.apply_to_metadata_rows(
        [True] * len(target),
        _repair_parent_mass_from_smiles_metadata,
        mass_tolerance=mass_tolerance,
    )

    return target


repair_parent_mass_from_smiles = collection_filter(
    _repair_parent_mass_from_smiles_spectrum,
    collection_impl=_repair_parent_mass_from_smiles_collection,
)
