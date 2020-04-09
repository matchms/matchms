from .add_adduct import add_adduct
from .correct_charge import correct_charge
from .derive_ionmode import derive_ionmode
from .make_charge_scalar import make_charge_scalar
from .make_ionmode_lowercase import make_ionmode_lowercase
from .set_ionmode_na_when_missing import set_ionmode_na_when_missing


def default_filters(s):
    make_charge_scalar(s)
    make_ionmode_lowercase(s)
    set_ionmode_na_when_missing(s)
    add_adduct(s)
    derive_ionmode(s)
    correct_charge(s)
