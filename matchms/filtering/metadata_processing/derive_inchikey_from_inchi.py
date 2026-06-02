import logging
from matchms import SpectraCollection
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.metadata_conversions import (
    apply_metadata_row_filter,
    as_string_or_none,
)
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    convert_inchi_to_inchikey,
    is_valid_inchi,
    is_valid_inchikey,
)
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _derive_inchikey_from_inchi(metadata) -> dict:
    """Return metadata updates with InChIKey derived from InChI where possible."""
    inchi = as_string_or_none(metadata.get("inchi"))
    inchikey = as_string_or_none(metadata.get("inchikey"))

    if not is_valid_inchi(inchi) or is_valid_inchikey(inchikey):
        return {}

    inchikey = convert_inchi_to_inchikey(inchi)
    if not inchikey:
        logger.warning("Could not convert InChI %s to inchikey.", inchi)
        return {}

    logger.info("Added InChIKey %s to metadata (was converted from InChI)", inchikey)

    return {"inchikey": inchikey}


def _derive_inchikey_from_inchi_spectrum(
    spectrum_in: SpectrumType,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Find missing InChIKey and derive from InChI where possible."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    updates = _derive_inchikey_from_inchi(spectrum.metadata)
    for key, value in updates.items():
        spectrum.set(key, value)

    return spectrum


def _derive_inchikey_from_inchi_collection(
    spectrum_in: SpectraCollection,
    clone: bool | None = True,
) -> SpectraCollection:
    """Find missing InChIKey and derive from InChI where possible for a collection."""
    target = spectrum_in.copy() if clone else spectrum_in

    target.apply_to_metadata_rows(
        apply_metadata_row_filter,
        row_filter=_derive_inchikey_from_inchi,
        inplace=True,
    )

    return target


derive_inchikey_from_inchi = collection_filter(
    _derive_inchikey_from_inchi_spectrum,
    collection_impl=_derive_inchikey_from_inchi_collection,
)
