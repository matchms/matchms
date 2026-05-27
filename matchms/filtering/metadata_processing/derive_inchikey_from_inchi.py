import logging
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.metadata_conversions import (
    derive_metadata_column_from_column,
)
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    convert_inchi_to_inchikey,
    is_valid_inchi,
    is_valid_inchikey,
)
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _derive_inchikey_from_inchi_spectrum(
    spectrum_in: SpectrumType,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Find missing InChIKey and derive from InChI where possible."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in
    inchi = spectrum.get("inchi")
    inchikey = spectrum.get("inchikey")

    if is_valid_inchi(inchi) and not is_valid_inchikey(inchikey):
        inchikey = convert_inchi_to_inchikey(inchi)
        if inchikey:
            spectrum.set("inchikey", inchikey)
            logger.info("Added InChIKey %s to metadata (was converted from InChI)", inchikey)
        else:
            logger.warning("Could not convert InChI %s to inchikey.", inchi)

    return spectrum


def _derive_inchikey_from_inchi_collection(
    spectrum_in,
    clone: bool | None = True,
):
    """Find missing InChIKey and derive from InChI where possible for a collection."""
    return derive_metadata_column_from_column(
        spectrum_in,
        source_key="inchi",
        target_key="inchikey",
        is_valid_source=is_valid_inchi,
        is_valid_target=is_valid_inchikey,
        converter=convert_inchi_to_inchikey,
        clone=clone,
        source_label="InChI",
        target_label="InChIKey",
    )


derive_inchikey_from_inchi = collection_filter(
    _derive_inchikey_from_inchi_spectrum,
    collection_impl=_derive_inchikey_from_inchi_collection,
)
