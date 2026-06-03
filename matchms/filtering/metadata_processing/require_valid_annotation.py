import logging
from matchms.filtering._dispatch import metadata_requirement_filter
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    convert_inchi_to_inchikey,
    convert_smiles_to_inchi,
    is_valid_inchi,
    is_valid_inchikey,
    is_valid_smiles,
)


logger = logging.getLogger("matchms")


def _require_valid_annotation(metadata) -> bool:
    """Require valid and matching SMILES, InChI, and InChIKey annotations.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
    clone
        Optionally clone the input before applying the filter. If ``False``,
        the input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Spectrum input is returned unchanged if annotations are valid and
        matching, otherwise ``None``. SpectraCollection input is returned with
        invalid rows removed.
    """
    smiles = as_string_or_none(metadata.get("smiles"))
    inchi = as_string_or_none(metadata.get("inchi"))
    inchikey = as_string_or_none(metadata.get("inchikey"))

    if not is_valid_smiles(smiles):
        logger.info("Removed spectrum since smiles is not valid. Incorrect smiles = %s", smiles)
        return False

    if not is_valid_inchikey(inchikey):
        logger.info("Removed spectrum since inchikey is not valid. Incorrect inchikey = %s", inchikey)
        return False

    if not is_valid_inchi(inchi):
        logger.info("Removed spectrum since inchi is not valid. Incorrect inchi = %s", inchi)
        return False

    if not _check_smiles_inchi_inchikey_match(smiles, inchi, inchikey):
        logger.info(
            "Removed spectrum since smiles, inchi and inchikey do not match. "
            "Smiles = %s, inchi = %s, inchikey = %s",
            smiles,
            inchi,
            inchikey,
        )
        return False

    return True


def _check_smiles_inchi_inchikey_match(smiles, inchi, inchikey) -> bool:
    """Check if SMILES, InChI, and InChIKey match."""
    inchikey_from_inchi = convert_inchi_to_inchikey(inchi)
    if inchikey_from_inchi is None:
        return False

    if inchikey[:14] != inchikey_from_inchi[:14]:
        return False

    inchi_from_smiles = convert_smiles_to_inchi(smiles)
    if inchi_from_smiles is None:
        return False

    inchikey_from_smiles = convert_inchi_to_inchikey(inchi_from_smiles)
    if inchikey_from_smiles is None:
        return False

    if inchikey[:14] != inchikey_from_smiles[:14]:
        return False

    return True


require_valid_annotation = metadata_requirement_filter(_require_valid_annotation)