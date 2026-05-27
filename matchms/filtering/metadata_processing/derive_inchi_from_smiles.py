import logging
from matchms.filtering._dispatch import collection_filter
from matchms.filtering.filter_utils.metadata_conversions import (
    derive_metadata_column_from_column,
)
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    convert_smiles_to_inchi,
    is_valid_inchi,
    is_valid_smiles,
)
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _derive_inchi_from_smiles_spectrum(
    spectrum_in: SpectrumType,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Find missing InChI and derive from smiles where possible."""
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
    spectrum_in,
    clone: bool | None = True,
):
    """Find missing InChI and derive from smiles where possible for a collection."""
    return derive_metadata_column_from_column(
        spectrum_in,
        source_key="smiles",
        target_key="inchi",
        is_valid_source=is_valid_smiles,
        is_valid_target=is_valid_inchi,
        converter=convert_smiles_to_inchi,
        clone=clone,
        source_label="smiles",
        target_label="InChI",
    )


derive_inchi_from_smiles = collection_filter(
    _derive_inchi_from_smiles_spectrum,
    collection_impl=_derive_inchi_from_smiles_collection,
)
