from matchms.typing import SpectrumType
from matchms.filtering.filters.make_charge_int import MakeChargeInt


def make_charge_int(spectrum_in: SpectrumType) -> SpectrumType:
    """Convert charge field to integer (if possible)."""

    spectrum = MakeChargeInt().process(spectrum_in)
    return spectrum
