"""Backward-compatible SpectrumProcessor alias.

SpectrumProcessor has been replaced by SpectraProcessor.
"""

from deprecated import deprecated
from .spectra_processor import SpectraProcessor


@deprecated(
    version="1.1.0",
    reason="SpectrumProcessor has been replaced by SpectraProcessor. "
    "Use SpectraProcessor instead.",
)
class SpectrumProcessor(SpectraProcessor):
    """Deprecated alias for :class:`~matchms.filtering.SpectraProcessor`."""