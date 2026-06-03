from matchms.filtering._dispatch import metadata_requirement_filter
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import (
    get_monoisotopic_neutral_mass,
)
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    as_string_or_none,
)


def _require_parent_mass_match_smiles(metadata, mass_tolerance) -> bool:
    """Validate that parent mass matches the mass calculated from SMILES.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
    mass_tolerance
        Allowed absolute mass difference between ``parent_mass`` and the
        monoisotopic neutral mass calculated from ``smiles``.
    clone
        Optionally clone the input before applying the filter. If ``False``,
        the input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Spectrum input is returned unchanged if ``parent_mass`` matches the
        SMILES-derived mass, otherwise ``None``. SpectraCollection input is
        returned with non-matching rows removed.
    """
    return _check_smiles_and_parent_mass_match(
        smiles=metadata.get("smiles"),
        parent_mass=metadata.get("parent_mass"),
        mass_tolerance=mass_tolerance,
    )


def _check_smiles_and_parent_mass_match(smiles, parent_mass, mass_tolerance) -> bool:
    """Return True if SMILES and parent mass are matching."""
    smiles = as_string_or_none(smiles)
    parent_mass = as_float_or_none(parent_mass)

    smiles_mass = get_monoisotopic_neutral_mass(smiles)

    if smiles_mass is None or parent_mass is None:
        return False

    return abs(parent_mass - smiles_mass) < mass_tolerance


require_parent_mass_match_smiles = metadata_requirement_filter(
    _require_parent_mass_match_smiles
)