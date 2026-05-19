from typing import Optional
import pandas as pd
from matchms import SpectraCollection
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.SpeciesString import SpeciesString
from matchms.typing import SpectrumType


def _repair_species_values(inchi, inchiaux, inchikey, smiles) -> dict[str, str]:
    """Repair and assign inchi, inchikey, and smiles values."""
    cleaneds = [SpeciesString(s) for s in [inchi, inchiaux, inchikey, smiles]]

    inchis = [
        c.cleaned for c in cleaneds
        if c.target == "inchi" and c.cleaned != ""
    ]
    inchikeys = [
        c.cleaned for c in cleaneds
        if c.target == "inchikey" and c.cleaned != ""
    ]
    smiles_values = [
        c.cleaned for c in cleaneds
        if c.target == "smiles" and c.cleaned != ""
    ]

    return {
        "inchi": inchis[0] if len(inchis) > 0 else "",
        "inchikey": inchikeys[0] if len(inchikeys) > 0 else "",
        "smiles": smiles_values[0] if len(smiles_values) > 0 else "",
    }


def _repair_inchi_inchikey_smiles_spectrum(
    spectrum_in: SpectrumType,
    clone: Optional[bool] = True,
) -> Optional[SpectrumType]:
    """Check if inchi, inchikey, and smiles entries seem correct.

    Detect and correct if any of those entries clearly belongs into one of the
    other two fields, for example if an inchikey is found in the inchi field.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    repaired = _repair_species_values(
        spectrum.get("inchi", ""),
        spectrum.get("inchiaux", ""),
        spectrum.get("inchikey", ""),
        spectrum.get("smiles", ""),
    )

    spectrum.set("inchi", repaired["inchi"])
    spectrum.set("inchikey", repaired["inchikey"])
    spectrum.set("smiles", repaired["smiles"])

    return spectrum


def _repair_species_row(row: pd.Series) -> pd.Series:
    repaired = _repair_species_values(
        row.get("inchi", ""),
        row.get("inchiaux", ""),
        row.get("inchikey", ""),
        row.get("smiles", ""),
    )
    return pd.Series(repaired)


def _repair_inchi_inchikey_smiles_collection(
    collection: SpectraCollection,
    clone: Optional[bool] = True,
) -> SpectraCollection:
    target = collection.copy() if clone else collection
    metadata = target._metadata.copy()

    for column in ["inchi", "inchiaux", "inchikey", "smiles"]:
        if column not in metadata.columns:
            metadata[column] = ""

    repaired = metadata[
        ["inchi", "inchiaux", "inchikey", "smiles"]
    ].apply(_repair_species_row, axis=1)

    metadata["inchi"] = repaired["inchi"]
    metadata["inchikey"] = repaired["inchikey"]
    metadata["smiles"] = repaired["smiles"]

    target._metadata = metadata
    target._clear_cache(["metadata_hashes", "spectra_hashes"])

    return target


# wrapper
repair_inchi_inchikey_smiles = collection_filter(
    _repair_inchi_inchikey_smiles_spectrum,
    collection_impl=None,
)
