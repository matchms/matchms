from typing import Union
from matchms import Spectrum


def has_valid_inchikey(spectrum_in) -> Union[Spectrum, None]:
    """Return True if input spectrum has a valid inchikey string."""

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    empty_entry_types = [
        "N/A",
        "n/a",
        "n\a",
        "NA",
        0,
        "0",
        '""',
        "",
        "nodata"
    ]

    inchikey = spectrum.get("inchikey")
    if inchikey is None:
        return False

    if inchikey in empty_entry_types:
        return False

    return True
