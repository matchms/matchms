from matchms.filtering.filters.add_precursor_mz import AddPrecursorMz

def add_precursor_mz(spectrum_in):
    """Add precursor_mz to correct field and make it a float.

    For missing precursor_mz field: check if there is "pepmass"" entry instead.
    For string parsed as precursor_mz: convert to float.
    """

    spectrum = AddPrecursorMz().process(spectrum_in)
    return spectrum
