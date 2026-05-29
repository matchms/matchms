import itertools
import logging
import pandas as pd
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import (
    get_monoisotopic_neutral_mass,
)
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    as_string_or_none,
)
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    is_valid_smiles,
)
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _find_matching_salt_ion(smiles, parent_mass, mass_tolerance):
    """Return matching ion and removed ions if one salt part matches parent mass."""
    smiles = as_string_or_none(smiles)
    parent_mass = as_float_or_none(parent_mass)

    if smiles is None or parent_mass is None:
        return None, None

    if not is_valid_smiles(smiles):
        return None, None

    possible_ion_combinations = _create_possible_ions(smiles)
    if not possible_ion_combinations:
        return None, None

    for ion, not_used_ions in possible_ion_combinations:
        ion_mass = get_monoisotopic_neutral_mass(ion)
        if ion_mass is None:
            continue

        mass_diff = abs(parent_mass - ion_mass)
        if mass_diff < mass_tolerance:
            return ion, not_used_ions

    logger.warning(
        "None of the parts of the smile %s match the parent mass: %s",
        smiles,
        parent_mass,
    )
    return None, None


def _repair_smiles_of_salts_metadata(
    metadata: pd.DataFrame,
    mass_tolerance,
) -> pd.DataFrame:
    """Repair salt smiles entries where one salt component matches parent_mass."""
    metadata = metadata.copy()

    if "smiles" not in metadata.columns or "parent_mass" not in metadata.columns:
        return metadata

    metadata["smiles"] = metadata["smiles"].astype("object")

    if "salt_ions" not in metadata.columns:
        metadata["salt_ions"] = None

    metadata["salt_ions"] = metadata["salt_ions"].astype("object")

    def _repair_row(row):
        smiles = row.get("smiles")
        parent_mass = row.get("parent_mass")

        ion, not_used_ions = _find_matching_salt_ion(
            smiles=smiles,
            parent_mass=parent_mass,
            mass_tolerance=mass_tolerance,
        )

        if ion is None:
            return pd.Series(
                {
                    "smiles": row.get("smiles"),
                    "salt_ions": row.get("salt_ions"),
                }
            )

        logger.info(
            "Removed salt ions: %s from %s to match parent mass",
            not_used_ions,
            smiles,
        )

        return pd.Series(
            {
                "smiles": ion,
                "salt_ions": not_used_ions,
            }
        )

    repaired = metadata.apply(_repair_row, axis=1)

    metadata["smiles"] = repaired["smiles"]
    metadata["salt_ions"] = repaired["salt_ions"]

    return metadata


def _repair_smiles_of_salts_spectrum(
    spectrum_in,
    mass_tolerance,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Repair salt SMILES to match parent mass."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    ion, not_used_ions = _find_matching_salt_ion(
        smiles=spectrum.get("smiles"),
        parent_mass=spectrum.get("parent_mass"),
        mass_tolerance=mass_tolerance,
    )

    if ion is None:
        return spectrum

    smiles = spectrum.get("smiles")
    spectrum.set("smiles", ion)
    spectrum.set("salt_ions", not_used_ions)

    logger.info(
        "Removed salt ions: %s from %s to match parent mass",
        not_used_ions,
        smiles,
    )

    return spectrum


def _repair_smiles_of_salts_collection(
    spectrum_in: SpectraCollection,
    mass_tolerance,
    clone: bool | None = True,
) -> SpectraCollection:
    """Repair salt SMILES to match parent mass for a SpectraCollection."""
    target = spectrum_in.copy() if clone else spectrum_in

    target = target.apply_to_metadata_rows(
        [True] * len(target),
        _repair_smiles_of_salts_metadata,
        mass_tolerance=mass_tolerance,
    )

    return target


def _create_possible_ions(smiles):
    """Select all possible ion combinations of a salt."""
    results = []

    if "." in smiles:
        single_ions = smiles.split(".")
        for r in range(1, len(single_ions) + 1):
            combinations = itertools.combinations(single_ions, r)
            for combination in combinations:
                combined_ion = ".".join(combination)
                removed_ions = single_ions.copy()
                for used_ion in combination:
                    removed_ions.remove(used_ion)
                removed_ions = ".".join(removed_ions)
                results.append((combined_ion, removed_ions))

    return results


repair_smiles_of_salts = collection_filter(
    _repair_smiles_of_salts_spectrum,
    collection_impl=_repair_smiles_of_salts_collection,
)
