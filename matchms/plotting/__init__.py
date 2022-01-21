"""
Functions for plotting one or multiple spectra
##############################################

Matchms provides (limited) plotting functionality to display one or multiple
spectra from :class:`~matchms.Spectrum` objects.

Currently this includes 3 different plot types:

* `plot_spectrum(spectrum)` or `spectrum.plot()` 
  This will create a plot of an individual spectrum.
* `plot_spectra_mirror(spectrum1, spectrum2)` or `spectrum1.plot_against(spectrum2)` 
  This will create a mirro plot comparing two spectra.
* `plot_spectra_array([spec1, spec2, ...])`
  This will create a plot with an array of all spectra in the given list.

Example of how to visually compare two spectra:

.. testcode::

    import numpy as np
    from matchms import Spectrum

    spectrum = Spectrum(mz=np.array([100, 120, 150, 200.]),
                        intensities=np.array([200.0, 300.0, 50.0, 45.0]),
                        metadata={'compound_name': 'spectrum1'})
    spectrum2 = Spectrum(mz=np.array([110, 130, 150, 200.]),
                        intensities=np.array([180.0, 250.0, 80.0, 30.0]),
                        metadata={'compound_name': 'spectrum2'})

    spectrum.plot_against(spectrum2)

.. figure:: ../_static/spectrum-plot-example.png
   :width: 700
   :alt: matchms spctrum plot

   Plot of individual spectrum.

"""
from .spectrum_plots import plot_spectra_array
from .spectrum_plots import plot_spectra_mirror
from .spectrum_plots import plot_spectrum


__all__ = [
    "plot_spectrum",
    "plot_spectra_mirror",
    "plot_spectra_array",
]
