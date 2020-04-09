from matchms.filtering import (add_adduct, derive_ionmode, correct_charge, make_ionmode_lowercase,
                               make_charge_scalar, set_ionmode_na_when_missing)


def default_filters(s):
    make_charge_scalar(s)
    make_ionmode_lowercase(s)
    set_ionmode_na_when_missing(s)
    add_adduct(s)
    derive_ionmode(s)
    correct_charge(s)
