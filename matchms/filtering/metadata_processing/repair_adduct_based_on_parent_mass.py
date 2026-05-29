import logging
import pandas as pd
from matchms import SpectraCollection, Spectrum
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    as_string_or_none,
)
from matchms.typing import SpectrumType
from ..filter_utils.load_known_adducts import load_known_adducts


logger = logging.getLogger("matchms")


def _repair_adduct_based_on_parent_mass_value(
    precursor_mz,
    parent_mass,
    ionmode,
    current_adduct,
    mass_tolerance,
):
    """Return repaired adduct if precursor_mz, parent_mass, and ionmode match a known adduct."""
    new_adduct = _get_matching_adduct(
        precursor_mz=precursor_mz,
        parent_mass=parent_mass,
        ion_mode=ionmode,
        mass_tolerance=mass_tolerance,
    )

    if new_adduct is None:
        return current_adduct

    if new_adduct != current_adduct:
        logger.info("Adduct was set from %s to %s", current_adduct, new_adduct)

    return new_adduct


def _repair_adduct_based_on_parent_mass_metadata(
    metadata: pd.DataFrame,
    mass_tolerance: float,
) -> pd.DataFrame:
    """Repair adduct column based on precursor_mz, parent_mass, and ionmode."""
    metadata = metadata.copy()

    required_columns = {"precursor_mz", "parent_mass", "ionmode"}
    if not required_columns.issubset(metadata.columns):
        return metadata

    if "adduct" not in metadata.columns:
        metadata["adduct"] = None

    metadata["adduct"] = metadata["adduct"].astype("object")

    metadata["adduct"] = metadata.apply(
        lambda row: _repair_adduct_based_on_parent_mass_value(
            precursor_mz=row.get("precursor_mz"),
            parent_mass=row.get("parent_mass"),
            ionmode=row.get("ionmode"),
            current_adduct=row.get("adduct"),
            mass_tolerance=mass_tolerance,
        ),
        axis=1,
    )

    return metadata


def _repair_adduct_based_on_parent_mass_spectrum(
    spectrum_in: Spectrum,
    mass_tolerance: float,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Correct adduct based on parent_mass and precursor_mz."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    repaired_adduct = _repair_adduct_based_on_parent_mass_value(
        precursor_mz=spectrum.get("precursor_mz"),
        parent_mass=spectrum.get("parent_mass"),
        ionmode=spectrum.get("ionmode"),
        current_adduct=spectrum.get("adduct"),
        mass_tolerance=mass_tolerance,
    )

    if repaired_adduct is not None:
        spectrum.set("adduct", repaired_adduct)

    return spectrum


def _repair_adduct_based_on_parent_mass_collection(
    spectrum_in: SpectraCollection,
    mass_tolerance: float,
    clone: bool | None = True,
) -> SpectraCollection:
    """Correct adducts based on parent_mass and precursor_mz for a SpectraCollection."""
    target = spectrum_in.copy() if clone else spectrum_in

    return target.apply_to_metadata_rows(
        [True] * len(target),
        _repair_adduct_based_on_parent_mass_metadata,
        mass_tolerance=mass_tolerance,
    )


def _get_matching_adduct(precursor_mz, parent_mass, ion_mode, mass_tolerance):
    precursor_mz = as_float_or_none(precursor_mz)
    parent_mass = as_float_or_none(parent_mass)
    ion_mode = as_string_or_none(ion_mode)

    if precursor_mz is None:
        logger.warning("Precursor_mz is None, first run add_precursor_mz")
        return None

    if ion_mode not in ("positive", "negative"):
        if ion_mode is not None:
            logger.warning(
                "Ionmode: %s not positive, negative or None, first run derive_ionmode",
                ion_mode,
            )
        return None

    if parent_mass is None:
        return None

    adducts_df = load_known_adducts()
    adducts_df = adducts_df[adducts_df["ionmode"] == ion_mode]

    # Do not use M+ and M- because these could accidentally repair cases where
    # parent_mass was filled into precursor_mz.
    adducts_df = adducts_df[~adducts_df["adduct"].isin(("[M]+", "[M]-"))]

    parent_masses = (
        precursor_mz - adducts_df["correction_mass"]
    ) / adducts_df["mass_multiplier"]
    mass_differences = abs(parent_masses - parent_mass)

    smallest_mass_index = mass_differences.idxmin()
    adduct = adducts_df.loc[smallest_mass_index]["adduct"]

    if mass_differences[smallest_mass_index] < mass_tolerance:
        return adduct
    return None


repair_adduct_based_on_parent_mass = collection_filter(
    _repair_adduct_based_on_parent_mass_spectrum,
    collection_impl=_repair_adduct_based_on_parent_mass_collection,
)
