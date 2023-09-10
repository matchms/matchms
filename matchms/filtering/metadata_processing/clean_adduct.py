from matchms.filtering.filters.clean_adduct import CleanAdduct


def clean_adduct(spectrum_in):
    """Clean adduct and make it consistent in style.
    Will transform adduct strings of type 'M+H+' to '[M+H]+'.

    Parameters
    ----------
    spectrum_in
        Matchms Spectrum object.
    """

    spectrum = CleanAdduct().process(spectrum_in)
    return spectrum
