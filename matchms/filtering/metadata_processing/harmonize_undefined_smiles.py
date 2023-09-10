from typing import List
from matchms.typing import SpectrumType
from matchms.filtering.filters.harmonize_undefined_smiles import HarmonizeUndefinedSmiles


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
    
    spectrum = HarmonizeUndefinedSmiles(undefined, aliases).process(spectrum_in)
    return spectrum