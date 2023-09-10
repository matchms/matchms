from ...metadata_utils import convert_smiles_to_inchi, is_valid_inchi, is_valid_smiles
from matchms.filtering.filters.derive_from_inchi_smile_template import DeriveFromInchiSmileTemplate


class DeriveInchiFromSmiles(DeriveFromInchiSmileTemplate):
    def __init__(self):
        super().__init__(
            derive_to="inchi",
            derive_from="smiles",
            convert_function=convert_smiles_to_inchi,
            log_message="Added InChI (%s) to metadata (was converted from smiles).",
        )

    def is_valid(self, derive_to, derive_from):
        return not is_valid_inchi(derive_to) and is_valid_smiles(derive_from)