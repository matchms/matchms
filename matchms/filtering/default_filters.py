from ..typing import SpectrumType
from .correct_charge import correct_charge
from .derive_adduct_from_name import derive_adduct_from_name
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
    spectrum = derive_adduct_from_name(spectrum)
    spectrum = derive_ionmode(spectrum)
    spectrum = correct_charge(spectrum)
    return spectrum
