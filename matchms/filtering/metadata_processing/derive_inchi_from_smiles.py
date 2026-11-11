import logging
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    convert_smiles_to_inchi,
    is_valid_inchi,
    is_valid_smiles,
)


logger = logging.getLogger("matchms")


def _derive_inchi_from_smiles(metadata) -> dict:
    """Find missing InChI and derive from smiles where possible."""
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


derive_inchi_from_smiles = metadata_update_filter(_derive_inchi_from_smiles)
