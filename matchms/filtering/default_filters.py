from matchms.typing import SpectrumType
import matchms.filtering as ms_filtering


def default_filters(spectrum: SpectrumType) -> SpectrumType:
    """
    Collection of filters that are considered default and that do no require any (factory) arguments.

    Collection is

    1. :meth:`~matchms.filtering.make_charge_int`
    2. :meth:`~matchms.filtering.add_compound_name`
    3. :meth:`~matchms.filtering.derive_adduct_from_name`
    4. :meth:`~matchms.filtering.derive_formula_from_name`
    5. :meth:`~matchms.filtering.clean_compound_name`
    6. :meth:`~matchms.filtering.interpret_pepmass`
    7. :meth:`~matchms.filtering.add_precursor_mz`
    8. :meth:`~matchms.filtering.derive_ionmode`
    9. :meth:`~matchms.filtering.correct_charge`

    """
    spectrum = ms_filtering.make_charge_int(spectrum)
    spectrum = ms_filtering.add_compound_name(spectrum)
    spectrum = ms_filtering.derive_adduct_from_name(spectrum)
    spectrum = ms_filtering.derive_formula_from_name(spectrum)
    spectrum = ms_filtering.clean_compound_name(spectrum)
    spectrum = ms_filtering.interpret_pepmass(spectrum)
    spectrum = ms_filtering.add_precursor_mz(spectrum)
    spectrum = ms_filtering.derive_ionmode(spectrum)
    spectrum = ms_filtering.correct_charge(spectrum)
    return spectrum
