from matchms.typing import SpectrumType


def has_valid_inchi(spectrum_in: SpectrumType) -> SpectrumType:

    """Return True if input is a valid inchi string."""
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
        '"InChI=n/a"',
        '"InChI="',
        "InChI=1S/N\n",
        "\t\r\n"
    ]

    inchi = spectrum.get("inchi")

    if inchi is None:
        return False

    if inchi in empty_entry_types:
        return False

    if len(inchi) < 12:
        return False

    return True
