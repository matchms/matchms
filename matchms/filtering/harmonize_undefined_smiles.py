from typing import List
from ..typing import SpectrumType


def harmonize_undefined_smiles(spectrum_in: SpectrumType, undefined: str = "",
                               aliases: List[str] = None) -> SpectrumType:
    """Replace all aliases for empty/undefined smiles entries by ``undefined``.

    Parameters
    ----------
    undefined:
        Give desired entry for undefined smiles fields. Default is "".
    aliases:
        Enter list of strings that are expected to represent undefined entries.
        Default is ["", "N/A", "NA", "n/a", "no data"].
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if aliases is None:
        aliases = [
            "",
            "N/A",
            "NA",
            "n/a",
            "no data"
        ]

    smiles = spectrum.get("smiles")
    if smiles is None:
        # spectrum does not have a "smiles" key in its metadata
        spectrum.set("smiles", undefined)
        return spectrum

    if smiles in aliases:
        # harmonize aliases for undefined values
        spectrum.set("smiles", undefined)

    return spectrum
