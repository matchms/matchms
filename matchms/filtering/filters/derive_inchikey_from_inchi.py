from ...metadata_utils import convert_inchi_to_inchikey, is_valid_inchi, is_valid_inchikey
from matchms.filtering.filters.derive_from_inchi_smile_template import DeriveFromInchiSmileTemplate


class DeriveInchikeyFromInchi(DeriveFromInchiSmileTemplate):
    def __init__(self):
        super().__init__(
            derive_to="inchikey",
            derive_from="inchi",
            convert_function=convert_inchi_to_inchikey,
            log_message="Added InChIKey %s to metadata (was converted from inchi)",
        )

    def is_valid(self, derive_to, derive_from):
        return is_valid_inchi(derive_from) and not is_valid_inchikey(derive_to)