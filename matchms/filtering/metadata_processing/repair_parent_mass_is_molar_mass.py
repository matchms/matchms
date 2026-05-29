import logging
import pandas as pd
from matchms import Spectrum
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import (
    get_molecular_weight_neutral_mass,
    get_monoisotopic_neutral_mass,
)
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    as_string_or_none,
)
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _repair_parent_mass_is_molar_mass_value(smiles, parent_mass, mass_tolerance):
    """Return monoisotopic mass if parent_mass seems to be molar mass."""
    smiles = as_string_or_none(smiles)
    parent_mass_float = as_float_or_none(parent_mass)

    if smiles is None or parent_mass_float is None:
        return parent_mass

    smiles_molecular_weight = get_molecular_weight_neutral_mass(smiles)
    if smiles_molecular_weight is None:
        return parent_mass

    mass_difference = parent_mass_float - smiles_molecular_weight
    if abs(mass_difference) > mass_tolerance:
        return parent_mass

    correct_mass = get_monoisotopic_neutral_mass(smiles)
    if correct_mass is None:
        return parent_mass

    logger.info(
        "Parent mass was molar mass instead of monoisotopic mass corrected from %s to %s",
        parent_mass_float,
        correct_mass,
    )
    return correct_mass


def _repair_parent_mass_is_molar_mass_metadata(
    metadata: pd.DataFrame,
    mass_tolerance: float,
) -> pd.DataFrame:
    """Change parent_mass from molar mass into monoisotopic mass where applicable."""
    metadata = metadata.copy()

    if "smiles" not in metadata.columns or "parent_mass" not in metadata.columns:
        return metadata

    metadata["parent_mass"] = metadata["parent_mass"].astype("object")

    metadata["parent_mass"] = metadata.apply(
        lambda row: _repair_parent_mass_is_molar_mass_value(
            smiles=row.get("smiles"),
            parent_mass=row.get("parent_mass"),
            mass_tolerance=mass_tolerance,
        ),
        axis=1,
    )

    return metadata


def _repair_parent_mass_is_molar_mass_spectrum(
    spectrum_in: Spectrum,
    mass_tolerance: float,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Change parent mass from molar mass into monoisotopic mass."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    repaired_parent_mass = _repair_parent_mass_is_molar_mass_value(
        smiles=spectrum.get("smiles"),
        parent_mass=spectrum.get("parent_mass"),
        mass_tolerance=mass_tolerance,
    )

    spectrum.set("parent_mass", repaired_parent_mass)

    return spectrum


def _repair_parent_mass_is_molar_mass_collection(
    spectrum_in: SpectraCollection,
    mass_tolerance: float,
    clone: bool | None = True,
) -> SpectraCollection:
    """Change parent mass from molar mass into monoisotopic mass for a SpectraCollection."""
    target = spectrum_in.copy() if clone else spectrum_in

    target = target.apply_to_metadata_rows(
        [True] * len(target),
        _repair_parent_mass_is_molar_mass_metadata,
        mass_tolerance=mass_tolerance,
    )

    return target


repair_parent_mass_is_molar_mass = collection_filter(
    _repair_parent_mass_is_molar_mass_spectrum,
    collection_impl=_repair_parent_mass_is_molar_mass_collection,
)
