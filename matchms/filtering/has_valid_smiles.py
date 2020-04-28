from ..typing import SpectrumType


def has_valid_smiles(spectrum_in: SpectrumType) -> SpectrumType:
    """Return True if input is a valid "smiles" string."""

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

    smiles = spectrum.get("smiles")
    if smiles is None:
        return False

    if smiles in empty_entry_types:
        return False

    return True
