from typing import List
from ..typing import SpectrumType


def harmonize_undefined_inchi(spectrum_in: SpectrumType, undefined: str = "",
                              aliases: List[str] = None) -> SpectrumType:
    """Replace all aliases for empty/undefined inchi entries by value of ``undefined`` argument.

    Parameters
    ----------
    undefined:
        Give desired entry for undefined inchi fields. Default is "".
    aliases:
        Enter list of strings that are expected to represent undefined entries.
        Default is ["", "N/A", "NA", "n/a"].
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if aliases is None:
        aliases = [
            "",
            "N/A",
            "NA",
            "n/a"
        ]

    inchi = spectrum.get("inchi")
    if inchi is None:
        # spectrum does not have an "inchi" key in its metadata
        spectrum.set("inchi", undefined)
        return spectrum

    if inchi in aliases:
        # harmonize aliases for undefined values
        spectrum.set("inchi", undefined)

    return spectrum
