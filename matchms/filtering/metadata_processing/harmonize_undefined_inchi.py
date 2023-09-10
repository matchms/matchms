from typing import List
from matchms.typing import SpectrumType
from matchms.filtering.filters.harmonize_undefined_inchi import HarmonizeUndefinedInchi


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

    spectrum = HarmonizeUndefinedInchi(undefined, aliases).process(spectrum_in)
    return spectrum