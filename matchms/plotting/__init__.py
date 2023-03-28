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

.. code-block:: python

    import os
    from matplotlib import pyplot as plt
    import matchms.filtering as msfilters
    from matchms.importing import load_from_msp

    module_root = os.getcwd()
    spectrums_file = os.path.join(module_root, "matchms", "tests", "MoNA-export-GC-MS-first10.msp")
    spectrums = list(load_from_msp(spectrums_file))
    spectrums = [msfilters.default_filters(s) for s in spectrums]

    spectrums[1].plot()
    # plt.savefig("spectrum-plot-example_1.png", dpi=300)  # If you want to save a plot

.. figure:: ../_static/spectrum-plot-example.png
   :width: 700
   :alt: matchms spectrum plot

   Plot of individual spectrum.

Another example is to compare two spectra visually using a mirror plot:

.. code-block:: python

    spectrums[2].plot_against(spectrums[3])
    plt.xlim(0, 200)

.. figure:: ../_static/spectrum-mirror-plot-example.png
   :width: 700
   :alt: matchms spectrum mirror plot

   Compare two spectra visually using a mirror plot.

Finally, it is also possible to plot many spectra at once using `plot_spectra_array()`:

.. code-block:: python

    from matchms.plotting import plot_spectra_array
    plot_spectra_array(spectrums[:4])

.. figure:: ../_static/spectra-array-plot-example.png
   :width: 700
   :alt: matchms spectra array plot

   Compare many spectra visually using an array plot.

"""
from .spectrum_plots import (plot_spectra_array, plot_spectra_mirror,
                             plot_spectrum)


__all__ = [
    "plot_spectrum",
    "plot_spectra_mirror",
    "plot_spectra_array",
]
