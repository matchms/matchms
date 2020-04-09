import numpy
from scipy.optimize import curve_fit


def select_by_intensity_expfit(spectrum, n_bins=10):

    def calc_bin_edges_intensity():
        mz_max = spectrum.mz.max()
        return numpy.linspace(0, mz_max, n_bins + 1)

    def function(x, a0, alpha):
        return a0 * numpy.exp(-alpha * x)

    bin_edges = calc_bin_edges_intensity()
    hist, _ = numpy.histogram(spectrum.mz, bins=bin_edges)
    popt, _ = curve_fit(function,
                        bin_edges,
                        hist)

    print()

