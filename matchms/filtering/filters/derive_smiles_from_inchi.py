from ...metadata_utils import convert_inchi_to_smiles, is_valid_inchi, is_valid_smiles
from matchms.filtering.filters.derive_from_inchi_smile_template import DeriveFromInchiSmileTemplate


class DeriveSmilesFromInchi(DeriveFromInchiSmileTemplate):
    def __init__(self):
        super().__init__(
            derive_to="smiles",
            derive_from="inchi",
            convert_function=convert_inchi_to_smiles,
            log_message="Added smiles %s to metadata (was converted from InChI)",
        )

    def is_valid(self, derive_to, derive_from):
        return not is_valid_smiles(derive_to) and is_valid_inchi(derive_from)