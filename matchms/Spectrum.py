from matplotlib import pyplot as plt
import numpy
from scipy.optimize import curve_fit


class Spectrum:
    """An example docstring for a class."""
    def __init__(self, mz, intensities, metadata=None):
        """An example docstring for a constructor."""
        self.mz = mz
        self.intensities = intensities
        if metadata is None:
            self.metadata = dict()
        else:
            self.metadata = metadata

    def __eq__(self, other):
        return \
            numpy.all(self.mz == other.mz) and \
            numpy.all(self.intensities == other.intensities) and \
            self.metadata == other.metadata

    def clone(self):
        """Return a deepcopy of the spectrum instance."""
        return Spectrum(mz=self.mz,
                        intensities=self.intensities,
                        metadata=self.metadata)

    def plot(self, intensity_from=0.0, intensity_to=None, with_histogram=False, with_expfit=False):
        """An example docstring for a method."""

        def plot_histogram():
            """plot the histogram of intensity values as horizontal bars, aligned with the spectrum axes"""
            def calc_bin_edges_intensity():
                """calculate various properties of the histogram bins, given a range in intensity defined by
                'intensity_from' and 'intensity_to', assuming a number of bins equal to 100."""
                edges = numpy.linspace(intensity_from, intensity_to, n_bins + 1)
                lefts = edges[:-1]
                rights = edges[1:]
                middles = (lefts + rights) / 2
                widths = rights - lefts
                return edges, middles, widths

            def exponential_decay_function(x, init, decay_factor):
                """function describing exponential decay"""
                return init * numpy.power(1 - decay_factor, x)

            def plot_expfit():
                """fit an exponential decay function to the bars of the histogram and plot the fitted line
                on top of the histogram bars."""
                k = numpy.argmax(counts == counts.max())
                offset = bin_middles[k]
                x_fit = bin_middles[k:] - offset
                y_fit = counts[k:]
                selection = y_fit > 0
                x_fit_nozero = x_fit[selection]
                y_fit_nozero = y_fit[selection]
                lower_bounds = [counts.max() - 1e-10, 0]
                upper_bounds = [counts.max() + 1e-10, decay_factor_max]
                try:
                    popt, _ = curve_fit(exponential_decay_function,
                                        x_fit_nozero,
                                        y_fit_nozero,
                                        bounds=(lower_bounds, upper_bounds))
                except Exception as e:
                    print(e)
                    popt = lower_bounds, 0.1
                ax1_expfit = exponential_decay_function(x_fit, *popt)
                plt.plot(ax1_expfit, x_fit + offset, color="#F80", marker=".")

            bin_edges, bin_middles, bin_widths = calc_bin_edges_intensity()
            counts, _ = numpy.histogram(self.intensities, bins=bin_edges)
            histogram_ax.set_ylim(bottom=intensity_from, top=intensity_to)
            plt.barh(bin_middles, counts, height=bin_widths)
            plt.title("histogram (n_bins={0})".format(n_bins))
            plt.xlabel("count")
            if with_expfit:
                plot_expfit()

        def plot_spectrum():
            """plot mz v. intensity"""
            def make_stems():
                """calculate where the stems of the spectrum peaks are going to be"""
                x = numpy.empty([2, self.mz.size], dtype="float")
                y = numpy.empty_like(x)
                for i, mz in enumerate(self.mz):
                    x[0:2, i] = [mz, mz]
                    y[0:2, i] = [0, self.intensities[i]]
                return x, y

            spectrum_ax.set_ylim(bottom=intensity_from, top=intensity_to)
            x, y = make_stems()
            plt.plot(x, y, color="#444", linewidth=1.0, marker="")
            plt.title("Spectrum")
            plt.xlabel("M/z")
            plt.ylabel("intensity")

        if with_expfit:
            assert with_histogram, "When 'with_expfit' is True, 'with_histogram' should also be True."

        if intensity_to is None:
            intensity_to = self.intensities.max() * 1.05

        n_bins = 100
        decay_factor_max = 1.0
        fig = plt.figure()

        if with_histogram:
            spectrum_ax = fig.add_axes([0.2, 0.1, 0.5, 0.8])
            plot_spectrum()
            histogram_ax = fig.add_axes([0.72, 0.1, 0.2, 0.8])
            plot_histogram()
            histogram_ax.set_yticklabels([])
        else:
            spectrum_ax = fig.add_axes([0.2, 0.1, 0.7, 0.8])
            plot_spectrum()
            histogram_ax = None

        return fig

    def get(self, key, default=None):
        return self._metadata.get(key, default)

    def set(self, key, value):
        self._metadata[key] = value
        return self

    @property
    def mz(self):
        """getter method for mz private variable"""
        return self._mz.copy()

    @mz.setter
    def mz(self, value):
        """setter method for mz private variable"""
        self._mz = value

    @property
    def intensities(self):
        """getter method for intensities private variable"""
        return self._intensities.copy()

    @intensities.setter
    def intensities(self, value):
        """setter method for intensities private variable"""
        self._intensities = value

    @property
    def metadata(self):
        """getter method for metadata private variable"""
        return self._metadata.copy()

    @metadata.setter
    def metadata(self, value):
        """setter method for metadata private variable"""
        self._metadata = value
