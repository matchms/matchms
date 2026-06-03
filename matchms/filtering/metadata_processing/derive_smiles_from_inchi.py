import logging
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    convert_inchi_to_smiles,
    is_valid_inchi,
    is_valid_smiles,
)


logger = logging.getLogger("matchms")


def _derive_smiles_from_inchi(metadata) -> dict:
    """Find missing smiles and derive from InChI where possible."""
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


derive_smiles_from_inchi = metadata_update_filter(_derive_smiles_from_inchi)