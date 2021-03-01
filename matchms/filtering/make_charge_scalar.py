from deprecated.sphinx import deprecated
from ..typing import SpectrumType
from .make_charge_int import make_charge_int


@deprecated(version='0.8.2', reason="Use expanded make_charge_int() instead.")
def make_charge_scalar(spectrum_in: SpectrumType) -> SpectrumType:
    """Convert charge field to scalar (if necessary).

    Deprecated function, now replaced by make_charge_int().
    """

    return make_charge_int(spectrum_in)
