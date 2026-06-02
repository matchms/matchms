import logging
from matchms import SpectraCollection
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.metadata_conversions import (
    apply_metadata_row_filter,
    as_string_or_none,
)
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    convert_inchi_to_smiles,
    is_valid_inchi,
    is_valid_smiles,
)
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _derive_smiles_from_inchi(metadata) -> dict:
    """Return metadata updates with SMILES derived from InChI where possible."""
    inchi = as_string_or_none(metadata.get("inchi"))
    smiles = as_string_or_none(metadata.get("smiles"))

    if is_valid_smiles(smiles) or not is_valid_inchi(inchi):
        return {}

    smiles = convert_inchi_to_smiles(inchi)
    if not smiles:
        logger.warning("Could not convert InChI %s to smiles.", inchi)
        return {}

    smiles = smiles.rstrip()
    logger.info("Added smiles %s to metadata (was converted from InChI)", smiles)

    return {"smiles": smiles}


def _derive_smiles_from_inchi_spectrum(
    spectrum_in: SpectrumType,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Find missing smiles and derive from InChI where possible."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    updates = _derive_smiles_from_inchi(spectrum.metadata)
    for key, value in updates.items():
        spectrum.set(key, value)

    return spectrum


def _derive_smiles_from_inchi_collection(
    spectrum_in: SpectraCollection,
    clone: bool | None = True,
) -> SpectraCollection:
    """Find missing smiles and derive from InChI where possible for a collection."""
    target = spectrum_in.copy() if clone else spectrum_in

    target.apply_to_metadata_rows(
        apply_metadata_row_filter,
        row_filter=_derive_smiles_from_inchi,
        inplace=True,
    )

    return target


derive_smiles_from_inchi = collection_filter(
    _derive_smiles_from_inchi_spectrum,
    collection_impl=_derive_smiles_from_inchi_collection,
)
