from ..typing import SpectrumType


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
