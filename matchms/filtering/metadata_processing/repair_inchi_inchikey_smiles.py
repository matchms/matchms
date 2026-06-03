from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none
from matchms.filtering.SpeciesString import SpeciesString


def _repair_species_values(inchi, inchiaux, inchikey, smiles) -> dict[str, str]:
    """Repair and assign inchi, inchikey, and smiles values."""
    cleaneds = [SpeciesString(s) for s in [inchi, inchiaux, inchikey, smiles]]

    inchis = [
        c.cleaned
        for c in cleaneds
        if c.target == "inchi" and c.cleaned != ""
    ]
    inchikeys = [
        c.cleaned
        for c in cleaneds
        if c.target == "inchikey" and c.cleaned != ""
    ]
    smiles_values = [
        c.cleaned
        for c in cleaneds
        if c.target == "smiles" and c.cleaned != ""
    ]

    return {
        "inchi": inchis[0] if inchis else "",
        "inchikey": inchikeys[0] if inchikeys else "",
        "smiles": smiles_values[0] if smiles_values else "",
    }


def _metadata_value_or_empty_string(metadata, key: str) -> str:
    """Return metadata value as string, or empty string for missing values."""
    value = as_string_or_none(metadata.get(key))
    return "" if value is None else value


def _repair_inchi_inchikey_smiles(metadata) -> dict[str, str]:
    """Check if inchi, inchikey, and smiles entries seem correct.

    Detect and correct if any of those entries clearly belongs into one of the
    other two fields, for example if an inchikey is found in the inchi field.
    """
    return _repair_species_values(
        _metadata_value_or_empty_string(metadata, "inchi"),
        _metadata_value_or_empty_string(metadata, "inchiaux"),
        _metadata_value_or_empty_string(metadata, "inchikey"),
        _metadata_value_or_empty_string(metadata, "smiles"),
    )


repair_inchi_inchikey_smiles = metadata_update_filter(
    _repair_inchi_inchikey_smiles
)