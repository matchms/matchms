from typing import List
from matchms.typing import SpectrumType
from matchms.filtering.filters.harmonize_undefined_inchikey import HarmonizeUndefinedInchikey


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

    spectrum = HarmonizeUndefinedInchikey(undefined, aliases).process(spectrum_in)
    return spectrum