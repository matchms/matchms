from typing import List
from ..typing import SpectrumType


def harmonize_undefined_inchikey(spectrum_in: SpectrumType, undefined: str = "",
                                 aliases: List[str] = None) -> SpectrumType:
    """Replace all aliases for empty/undefined inchikey entries by ``undefined``.

    Parameters
    ----------
    undefined:
        Give desired entry for undefined inchikey fields. Default is "".
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

    inchikey = spectrum.get("inchikey")
    if inchikey is None:
        # spectrum does not have an "inchikey" key in its metadata
        spectrum.set("inchikey", undefined)
        return spectrum

    if inchikey in aliases:
        # harmonize aliases for undefined values
        spectrum.set("inchikey", undefined)

    return spectrum
