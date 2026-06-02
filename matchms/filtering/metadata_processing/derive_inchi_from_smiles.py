import logging
from matchms import SpectraCollection
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.metadata_conversions import (
    apply_metadata_row_filter,
    as_string_or_none,
)
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    convert_smiles_to_inchi,
    is_valid_inchi,
    is_valid_smiles,
)
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _derive_inchi_from_smiles(metadata, clone: bool | None = True) -> dict:
    """Return metadata updates with InChI derived from SMILES where possible."""
    inchi = as_string_or_none(metadata.get("inchi"))
    smiles = as_string_or_none(metadata.get("smiles"))

    if is_valid_inchi(inchi) or not is_valid_smiles(smiles):
        return {}

    inchi = convert_smiles_to_inchi(smiles)
    if not inchi:
        logger.warning("Could not convert smiles %s to InChI.", smiles)
        return {}

    inchi = inchi.rstrip()
    logger.info("Added InChI (%s) to metadata (was converted from smiles).", inchi)

    return {"inchi": inchi}


def _derive_inchi_from_smiles_spectrum(
    spectrum_in: SpectrumType,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Find missing InChI and derive from smiles where possible."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    updates = _derive_inchi_from_smiles(spectrum.metadata)
    for key, value in updates.items():
        spectrum.set(key, value)

    return spectrum


def _derive_inchi_from_smiles_collection(
    spectrum_in: SpectraCollection,
    clone: bool | None = True,
) -> SpectraCollection:
    """Find missing InChI and derive from smiles where possible for a collection."""
    target = spectrum_in.copy() if clone else spectrum_in

    target.apply_to_metadata_rows(
        apply_metadata_row_filter,
        row_filter=_derive_inchi_from_smiles,
        inplace=True,
    )

    return target


derive_inchi_from_smiles = collection_filter(
    _derive_inchi_from_smiles_spectrum,
    collection_impl=_derive_inchi_from_smiles_collection,
)
