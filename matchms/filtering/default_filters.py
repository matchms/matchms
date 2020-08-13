from ..typing import SpectrumType
from .add_compound_name import add_compound_name
from .add_precursor_mz import add_precursor_mz
from .clean_compound_name import clean_compound_name
from .correct_charge import correct_charge
from .derive_adduct_from_name import derive_adduct_from_name
from .derive_formula_from_name import derive_formula_from_name
from .derive_ionmode import derive_ionmode
from .make_charge_scalar import make_charge_scalar
from .make_ionmode_lowercase import make_ionmode_lowercase
from .set_ionmode_na_when_missing import set_ionmode_na_when_missing


def default_filters(spectrum: SpectrumType) -> SpectrumType:
    """
    Collection of filters that are considered default and that do no require any (factory) arguments.

    Collection is

    1. :meth:`~matchms.filtering.make_charge_scalar`
    2. :meth:`~matchms.filtering.make_ionmode_lowercase`
    3. :meth:`~matchms.filtering.set_ionmode_na_when_missing`
    4. :meth:`~matchms.filtering.add_compound_name`
    5. :meth:`~matchms.filtering.derive_adduct_from_name`
    6. :meth:`~matchms.filtering.derive_formula_from_name`
    7. :meth:`~matchms.filtering.clean_compound_name`
    8. :meth:`~matchms.filtering.add_precursor_mz`
    9. :meth:`~matchms.filtering.derive_ionmode`
    10. :meth:`~matchms.filtering.correct_charge`

    """
    spectrum = make_charge_scalar(spectrum)
    spectrum = make_ionmode_lowercase(spectrum)
    spectrum = set_ionmode_na_when_missing(spectrum)
    spectrum = add_compound_name(spectrum)
    spectrum = derive_adduct_from_name(spectrum)
    spectrum = derive_formula_from_name(spectrum)
    spectrum = clean_compound_name(spectrum)
    spectrum = add_precursor_mz(spectrum)
    spectrum = derive_ionmode(spectrum)
    spectrum = correct_charge(spectrum)
    return spectrum
