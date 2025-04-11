from matchms.typing import SpectrumType
from .metadata_processing.add_compound_name import add_compound_name
from .metadata_processing.add_precursor_mz import add_precursor_mz
from .metadata_processing.clean_compound_name import clean_compound_name
from .metadata_processing.correct_charge import correct_charge
from .metadata_processing.derive_adduct_from_name import derive_adduct_from_name
from .metadata_processing.derive_formula_from_name import derive_formula_from_name
from .metadata_processing.derive_ionmode import derive_ionmode
from .metadata_processing.interpret_pepmass import interpret_pepmass
from .metadata_processing.make_charge_int import make_charge_int


def default_filters(spectrum: SpectrumType) -> SpectrumType:
    """
    Collection of filters that are considered default and that do no require any (factory) arguments.

    Collection is

    1. :meth:`~matchms.filtering.metadata_processing.make_charge_int`
    2. :meth:`~matchms.filtering.metadata_processing.add_compound_name`
    3. :meth:`~matchms.filtering.metadata_processing.derive_adduct_from_name`
    4. :meth:`~matchms.filtering.metadata_processing.derive_formula_from_name`
    5. :meth:`~matchms.filtering.metadata_processing.clean_compound_name`
    6. :meth:`~matchms.filtering.metadata_processing.interpret_pepmass`
    7. :meth:`~matchms.filtering.metadata_processing.add_precursor_mz`
    8. :meth:`~matchms.filtering.metadata_processing.derive_ionmode`
    9. :meth:`~matchms.filtering.metadata_processing.correct_charge`

    """
    spectrum = make_charge_int(spectrum)
    spectrum = add_compound_name(spectrum)
    spectrum = derive_adduct_from_name(spectrum)
    spectrum = derive_formula_from_name(spectrum)
    spectrum = clean_compound_name(spectrum)
    spectrum = interpret_pepmass(spectrum)
    spectrum = add_precursor_mz(spectrum)
    spectrum = derive_ionmode(spectrum)
    spectrum = correct_charge(spectrum)
    return spectrum
