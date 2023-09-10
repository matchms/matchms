from matchms.filtering.filters.interpret_pepmass import InterpretPepmass


def interpret_pepmass(spectrum_in):
    """Reads pepmass field (if present) and adds values to correct field(s).

    The field "pepmass" or "PEPMASS" is often used to describe the precursor ion.
    This function will interpret the values as (mz, intensity, charge) tuple. Those
    will be splitted (if present) added to the fields "precursor_mz",
    "precursor_intensity", and "charge".
    """

    spectrum = InterpretPepmass().process(spectrum_in)
    return spectrum
