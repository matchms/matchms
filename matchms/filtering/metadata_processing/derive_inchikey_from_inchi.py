import logging
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    convert_inchi_to_inchikey,
    is_valid_inchi,
    is_valid_inchikey,
)
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _derive_inchikey_from_inchi_spectrum(
        spectrum_in: SpectrumType,
        clone: bool | None = True,
    ) -> SpectrumType | None:
    """Find missing InchiKey and derive from Inchi where possible.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with added INCHIKEY, or `None` if not present.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in
    inchi = spectrum.get("inchi")
    inchikey = spectrum.get("inchikey")

    if is_valid_inchi(inchi) and not is_valid_inchikey(inchikey):
        inchikey = convert_inchi_to_inchikey(inchi)
        if inchikey:
            spectrum.set("inchikey", inchikey)
            logger.info("Added InChIKey %s to metadata (was converted from inchi)", inchikey)
        else:
            logger.warning("Could not convert InChI %s to inchikey.", inchi)

    return spectrum


def _derive_inchikey_from_inchi_collection(
        spectrum_in: SpectraCollection,
        clone: bool | None = True,
    ) -> SpectraCollection:
    """Find missing InChIKey and derive from InChI where possible for a collection."""
    target = spectrum_in.copy() if clone else spectrum_in
    metadata = target._metadata.copy()

    if "inchi" not in metadata.columns:
        return target

    if "inchikey" not in metadata.columns:
        metadata["inchikey"] = None

    for idx, row in metadata.iterrows():
        inchi = row.get("inchi")
        inchikey = row.get("inchikey")

        if is_valid_inchi(inchi) and not is_valid_inchikey(inchikey):
            converted_inchikey = convert_inchi_to_inchikey(inchi)
            if converted_inchikey:
                metadata.at[idx, "inchikey"] = converted_inchikey
                logger.info(
                    "Added InChIKey %s to metadata (was converted from inchi)",
                    converted_inchikey,
                )
            else:
                logger.warning("Could not convert InChI %s to inchikey.", inchi)

    target._metadata = metadata
    target._clear_cache(["metadata_hashes", "spectra_hashes"])

    return target


derive_inchikey_from_inchi = collection_filter(
    _derive_inchikey_from_inchi_spectrum,
    collection_impl=_derive_inchikey_from_inchi_collection,
)
