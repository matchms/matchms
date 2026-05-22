import logging
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    convert_smiles_to_inchi,
    is_valid_inchi,
    is_valid_smiles,
)
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _derive_inchi_from_smiles_spectrum(
        spectrum_in: SpectrumType,
        clone: bool | None = True,
    ) -> SpectrumType | None:
    """Find missing Inchi and derive from smiles where possible.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with added INCHI, or `None` if not present.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in
    inchi = spectrum.get("inchi")
    smiles = spectrum.get("smiles")

    if not is_valid_inchi(inchi) and is_valid_smiles(smiles):
        inchi = convert_smiles_to_inchi(smiles)
        if inchi:
            inchi = inchi.rstrip()
            spectrum.set("inchi", inchi)
            logger.info("Added InChI (%s) to metadata (was converted from smiles).", inchi)
        else:
            logger.warning("Could not convert smiles %s to InChI.", smiles)

    return spectrum


def _derive_inchi_from_smiles_collection(
    spectrum_in: SpectraCollection,
    clone: bool | None = True,
) -> SpectraCollection:
    """Find missing InChI and derive from smiles where possible for a collection."""
    target = spectrum_in.copy() if clone else spectrum_in
    metadata = target._metadata.copy()

    if "smiles" not in metadata.columns:
        return target

    if "inchi" not in metadata.columns:
        metadata["inchi"] = None

    for idx, row in metadata.iterrows():
        inchi = row.get("inchi")
        smiles = row.get("smiles")

        if not is_valid_inchi(inchi) and is_valid_smiles(smiles):
            converted_inchi = convert_smiles_to_inchi(smiles)
            if converted_inchi:
                metadata.at[idx, "inchi"] = converted_inchi.rstrip()
                logger.info(
                    "Added InChI (%s) to metadata (was converted from smiles).",
                    converted_inchi,
                )
            else:
                logger.warning("Could not convert smiles %s to InChI.", smiles)

    target._metadata = metadata
    target._clear_cache(["metadata_hashes", "spectra_hashes"])

    return target


derive_inchi_from_smiles = collection_filter(
    _derive_inchi_from_smiles_spectrum,
    collection_impl=_derive_inchi_from_smiles_collection,
)
