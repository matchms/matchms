from ..typing import SpectrumType
from .add_adduct import add_adduct
from .add_precursor_mz import add_precursor_mz
from .correct_charge import correct_charge
from .derive_ionmode import derive_ionmode
from .make_charge_scalar import make_charge_scalar
from .make_ionmode_lowercase import make_ionmode_lowercase
from .set_ionmode_na_when_missing import set_ionmode_na_when_missing


def default_filters(spectrum: SpectrumType) -> SpectrumType:
    """
    Collection of filters that are considered default and that do no require any (factory) arguments.

    Collection is

    1. :meth:`~matchms.filtering.make_charge_scalar.make_charge_scalar`
    2. :meth:`~matchms.filtering.make_charge_scalar`
    3. :meth:`~matchms.filtering.make_ionmode_lowercase`
    4. :meth:`~matchms.filtering.set_ionmode_na_when_missing`
    5. :meth:`~matchms.filtering.add_precursor_mz`
    6. :meth:`~matchms.filtering.add_adduct`
    7. :meth:`~matchms.filtering.derive_ionmode`
    8. :meth:`~matchms.filtering.correct_charge`

    """
    spectrum = make_charge_scalar(spectrum)
    spectrum = make_ionmode_lowercase(spectrum)
    spectrum = set_ionmode_na_when_missing(spectrum)
    spectrum = add_precursor_mz(spectrum)
    spectrum = add_adduct(spectrum)
    spectrum = derive_ionmode(spectrum)
    spectrum = correct_charge(spectrum)
    return spectrum
