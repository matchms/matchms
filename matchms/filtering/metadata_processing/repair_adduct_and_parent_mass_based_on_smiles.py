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
from ..filter_utils.derive_precursor_mz_and_parent_mass import (
    derive_parent_mass_from_precursor_mz,
)
from ..filter_utils.load_known_adducts import load_known_adducts
from .repair_adduct_based_on_parent_mass import _get_matching_adduct


logger = logging.getLogger("matchms")


def _estimate_parent_mass_from_adduct_metadata(row):
    """Estimate parent mass from precursor_mz and adduct for metadata rows.

    This mirrors the spectrum-level derive_parent_mass_from_precursor_mz path,
    but avoids reconstructing Spectrum objects.
    """
    precursor_mz = as_float_or_none(row.get("precursor_mz"))
    adduct = as_string_or_none(row.get("adduct"))

    if precursor_mz is None or adduct is None:
        return None

    adducts_df = load_known_adducts()
    match = adducts_df[adducts_df["adduct"] == adduct]

    if match.empty:
        return None

    adduct_row = match.iloc[0]
    return (
        precursor_mz - adduct_row["correction_mass"]
    ) / adduct_row["mass_multiplier"]


def _repair_adduct_and_parent_mass_based_on_smiles_row(
    row,
    mass_tolerance: float,
):
    """Return repaired values for one metadata row."""
    smiles = as_string_or_none(row.get("smiles"))
    smiles_mass = get_monoisotopic_neutral_mass(smiles)

    if smiles_mass is None:
        return pd.Series(
            {
                "adduct": row.get("adduct"),
                "parent_mass": row.get("parent_mass"),
            }
        )

    parent_mass = as_float_or_none(row.get("parent_mass"))

    estimated_parent_mass = _estimate_parent_mass_from_adduct_metadata(row)

    need_to_update_adduct = False
    if estimated_parent_mass is not None:
        if abs(estimated_parent_mass - smiles_mass) > mass_tolerance:
            need_to_update_adduct = True
    else:
        need_to_update_adduct = True

    adduct = row.get("adduct")

    if need_to_update_adduct:
        new_adduct = _get_matching_adduct(
            precursor_mz=row.get("precursor_mz"),
            parent_mass=smiles_mass,
            ion_mode=row.get("ionmode"),
            mass_tolerance=mass_tolerance,
        )

        if new_adduct is None:
            return pd.Series(
                {
                    "adduct": row.get("adduct"),
                    "parent_mass": row.get("parent_mass"),
                }
            )

        logger.info("Adduct was set from %s to %s", row.get("adduct"), new_adduct)
        adduct = new_adduct

    repaired_parent_mass = row.get("parent_mass")

    if parent_mass is None:
        repaired_parent_mass = smiles_mass
        logger.info("Parent mass was set to match the smiles mass: %s", smiles_mass)
    elif abs(smiles_mass - parent_mass) > mass_tolerance:
        repaired_parent_mass = smiles_mass
        logger.info(
            "Parent mass was updated from %s to %s to match the smiles mass",
            parent_mass,
            smiles_mass,
        )

    return pd.Series(
        {
            "adduct": adduct,
            "parent_mass": repaired_parent_mass,
        }
    )


def _repair_adduct_and_parent_mass_based_on_smiles_metadata(
    metadata: pd.DataFrame,
    mass_tolerance: float,
) -> pd.DataFrame:
    """Repair adduct and parent_mass based on smiles and precursor_mz."""
    metadata = metadata.copy()

    if "smiles" not in metadata.columns:
        return metadata

    if "adduct" not in metadata.columns:
        metadata["adduct"] = None
    if "parent_mass" not in metadata.columns:
        metadata["parent_mass"] = None

    metadata["adduct"] = metadata["adduct"].astype("object")
    metadata["parent_mass"] = metadata["parent_mass"].astype("object")

    repaired = metadata.apply(
        _repair_adduct_and_parent_mass_based_on_smiles_row,
        axis=1,
        mass_tolerance=mass_tolerance,
    )

    metadata["adduct"] = repaired["adduct"]
    metadata["parent_mass"] = repaired["parent_mass"]

    return metadata


def _repair_adduct_and_parent_mass_based_on_smiles_spectrum(
    spectrum_in: Spectrum,
    mass_tolerance: float,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Corrects the adduct and parent mass of a spectrum based on its SMILES representation and the precursor m/z.

    Given a spectrum, this function tries to match the spectrum's parent mass, derived from its
    precursor m/z and known adducts, to the neutral monoisotopic mass of the molecule derived
    from its SMILES representation. If a match is found within a given mass tolerance, the
    adduct and parent mass of the spectrum are updated.

    Parameters:
    ----------
    spectrum_in : Spectrum
        The input spectrum whose adduct needs to be repaired.

    mass_tolerance : float
        Maximum allowed mass difference between the calculated parent mass and the neutral
        monoisotopic mass derived from the SMILES.

    clone:
        Optionally clone the Spectrum.
    """

    if spectrum_in is None:
        return None

    changed_spectrum = spectrum_in.clone() if clone else spectrum_in
    smiles_mass = get_monoisotopic_neutral_mass(changed_spectrum.get("smiles"))

    if smiles_mass is None:
        return changed_spectrum

    parent_mass = as_float_or_none(spectrum_in.get("parent_mass"))

    estimated_parent_mass = derive_parent_mass_from_precursor_mz(
        changed_spectrum,
        estimate_from_adduct=True,
        estimate_from_charge=False,
    )

    need_to_update_adduct = False
    if estimated_parent_mass is not None:
        if abs(estimated_parent_mass - smiles_mass) > mass_tolerance:
            need_to_update_adduct = True
    else:
        need_to_update_adduct = True

    if need_to_update_adduct:
        new_adduct = _get_matching_adduct(
            precursor_mz=spectrum_in.get("precursor_mz"),
            parent_mass=smiles_mass,
            ion_mode=spectrum_in.get("ionmode"),
            mass_tolerance=mass_tolerance,
        )

        if new_adduct is None:
            return changed_spectrum

        changed_spectrum.set("adduct", new_adduct)
        logger.info(
            "Adduct was set from %s to %s",
            spectrum_in.get("adduct"),
            new_adduct,
        )

    if parent_mass is None:
        changed_spectrum.set("parent_mass", smiles_mass)
        logger.info("Parent mass was set to match the smiles mass: %s", smiles_mass)
    elif abs(smiles_mass - parent_mass) > mass_tolerance:
        changed_spectrum.set("parent_mass", smiles_mass)
        logger.info(
            "Parent mass was updated from %s to %s to match the smiles mass",
            parent_mass,
            smiles_mass,
        )

    return changed_spectrum


def _repair_adduct_and_parent_mass_based_on_smiles_collection(
    spectrum_in: SpectraCollection,
    mass_tolerance: float,
    clone: bool | None = True,
) -> SpectraCollection:
    """Repair adduct and parent_mass based on smiles for a SpectraCollection."""
    target = spectrum_in.copy() if clone else spectrum_in

    return target.apply_to_metadata_rows(
        _repair_adduct_and_parent_mass_based_on_smiles_metadata,
        mass_tolerance=mass_tolerance,
    )


repair_adduct_and_parent_mass_based_on_smiles = collection_filter(
    _repair_adduct_and_parent_mass_based_on_smiles_spectrum,
    collection_impl=_repair_adduct_and_parent_mass_based_on_smiles_collection,
)
