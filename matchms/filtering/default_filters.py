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
    Collection of filters that are considered default and that do no require any parameterization.
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
