import logging
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.metadata_conversions import (
    derive_metadata_column_from_column,
)
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    convert_inchi_to_smiles,
    is_valid_inchi,
    is_valid_smiles,
)
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _derive_smiles_from_inchi_spectrum(
    spectrum_in: SpectrumType,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Find missing smiles and derive from InChI where possible."""
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
    spectrum_in,
    clone: bool | None = True,
):
    """Find missing smiles and derive from InChI where possible for a collection."""
    return derive_metadata_column_from_column(
        spectrum_in,
        source_key="inchi",
        target_key="smiles",
        is_valid_source=is_valid_inchi,
        is_valid_target=is_valid_smiles,
        converter=convert_inchi_to_smiles,
        clone=clone,
        source_label="InChI",
        target_label="smiles",
    )


derive_smiles_from_inchi = collection_filter(
    _derive_smiles_from_inchi_spectrum,
    collection_impl=_derive_smiles_from_inchi_collection,
)
