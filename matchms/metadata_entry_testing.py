from .typing import SpectrumType


def entry_is_empty(spectrum_in: SpectrumType,
                   metadata_field) -> bool:
    """Return True if input looks like an empty entry.

    Args:
    ----
    spectrum_in: matchms.Spectrum()
        Input spectrum object.
    metadata_field: str
        Specify metadata field to be tested for empty entries.
    """
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
        "nodata",
        "1S/N\n",
        "\t\r\n"
        "\r\n"
    ]

    entry = spectrum.get(metadata_field)

    if entry is None:
        return True

    if entry in empty_entry_types:
        return True

    if entry.split("InChI=")[-1] in empty_entry_types:
        return True

    return False


def is_valid_inchikey(inchikey):
    """Return True if input string has format of inchikey.

    This functions test if string has correct format of:
    "XXXXXXXXXXXXXXX-XXXXXXXXXX-X",
    with "X" being a letter of the alphabet.

    Args:
    ----
    inchikey: str
        Input string to test if it has format of inchikey.
    """
    if not isinstance(inchikey, str):
        return False

    # Harmonize string
    inchikey = inchikey.upper().replace('"', '').replace(' ', '')
    # Test if string looks like inchikey
    if not len(inchikey) == 27:
        return False

    if not inchikey[14] == inchikey[25] == "-":
        return False

    return sum([char.isalpha() for char in inchikey]) == 25
