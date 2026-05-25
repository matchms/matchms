import logging
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    _as_string_or_none,
    convert_inchi_to_smiles,
    is_valid_inchi,
    is_valid_smiles,
)
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _derive_smiles_from_inchi_spectrum(
        spectrum_in: SpectrumType,
        clone: bool | None = True,
    ) -> SpectrumType | None:
    """Find missing smiles and derive from Inchi where possible.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    clone:
        Optionally clone the Spectrum.#

    Returns
    -------
    Spectrum or None
        Spectrum with added SMILES, or `None` if not present.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in
    inchi = spectrum.get("inchi")
    smiles = spectrum.get("smiles")

    if not is_valid_smiles(smiles) and is_valid_inchi(inchi):
        smiles = convert_inchi_to_smiles(inchi)
        if smiles:
            smiles = smiles.rstrip()
            spectrum.set("smiles", smiles)
            logger.info("Added smiles %s to metadata (was converted from InChI)", smiles)
        else:
            logger.warning("Could not convert InChI %s to smiles.", inchi)

    return spectrum


def _derive_smiles_from_inchi_collection(
    spectrum_in: SpectraCollection,
    clone: bool | None = True,
) -> SpectraCollection:
    """Find missing smiles and derive from InChI where possible for a collection."""
    target = spectrum_in.copy() if clone else spectrum_in
    metadata = target._metadata.copy()

    if "inchi" not in metadata.columns:
        return target

    if "smiles" not in metadata.columns:
        metadata["smiles"] = None

    for idx, row in metadata.iterrows():
        inchi = _as_string_or_none(row.get("inchi"))
        smiles = _as_string_or_none(row.get("smiles"))

        if not is_valid_smiles(smiles) and is_valid_inchi(inchi):
            converted_smiles = convert_inchi_to_smiles(inchi)
            if converted_smiles:
                metadata.at[idx, "smiles"] = converted_smiles.rstrip()
                logger.info(
                    "Added smiles %s to metadata (was converted from InChI)",
                    converted_smiles,
                )
            else:
                logger.warning("Could not convert InChI %s to smiles.", inchi)

    target._metadata = metadata
    target._clear_cache(["metadata_hashes", "spectra_hashes"])

    return target


derive_smiles_from_inchi = collection_filter(
    _derive_smiles_from_inchi_spectrum,
    collection_impl=_derive_smiles_from_inchi_collection,
)
