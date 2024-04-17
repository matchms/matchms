import logging
import re
from typing import Optional
from matchms.typing import SpectrumType
from matchms.filtering.metadata_processing.derive_formula_from_name import _looks_like_formula


logger = logging.getLogger("matchms")


def require_formula(spectrum: SpectrumType) -> Optional[SpectrumType]:
    """Ensure that the molecular formula is present and looks like a valid formula."""
    if spectrum is None:
        return None

    formula = spectrum.get('formula', None)
    if formula and _is_valid_formula(formula):
        return spectrum
    else:
        logger.info("Spectrum does not contain a valid molecular formula.")
        return None


def _is_valid_formula(formula: str) -> bool:
    """Check if string looks like a valid chemical formula."""
    pattern = r'^([A-Z][a-z]?\d*)+$'
    return re.fullmatch(pattern, formula) is not None
