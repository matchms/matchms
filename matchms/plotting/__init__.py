"""
Functions for plotting one or multiple spectra
##############################################

"""
from .spectrum_plots import plot_spectra_array
from .spectrum_plots import plot_spectra_mirror
from .spectrum_plots import plot_spectrum


__all__ = [
    "plot_spectrum",
    "plot_spectra_mirror",
    "plot_spectra_array",
]
