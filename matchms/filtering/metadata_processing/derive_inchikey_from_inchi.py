import logging
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    convert_inchi_to_inchikey,
    is_valid_inchi,
    is_valid_inchikey,
)


logger = logging.getLogger("matchms")


def _derive_inchikey_from_inchi(metadata) -> dict:
    """Find missing InChIKey and derive from InChI where possible."""
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


derive_inchikey_from_inchi = metadata_update_filter(_derive_inchikey_from_inchi)