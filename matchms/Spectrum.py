from typing import Optional
import numpy as np
from matplotlib import pyplot
from .Fragments import Fragments
from .hashing import metadata_hash
from .hashing import spectrum_hash
from .Metadata import Metadata


class Spectrum:
    """Container for a collection of peaks, losses and metadata

    For example

    .. testcode::

        import numpy as np
        from matchms import Scores, Spectrum
        from matchms.similarity import CosineGreedy

        spectrum = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.7, 0.2, 0.1]),
                              metadata={'id': 'spectrum1',
                                        "peak_comments": {200.: "the peak at 200 m/z"}})

        print(spectrum.peaks.mz[0])
        print(spectrum.peaks.intensities[0])
        print(spectrum.get('id'))
        print(spectrum.peak_comments.get(200))

    Should output

    .. testoutput::

        100.0
        0.7
        spectrum1
        the peak at 200 m/z

    Attributes
    ----------
    peaks: ~matchms.Fragments.Fragments
        Peaks of spectrum
    losses: ~matchms.Fragments.Fragments or None
        Losses of spectrum, the difference between the precursor and all peaks.

        Can be filled with

        .. code-block ::

            from matchms import Fragments
            spectrum.losess = Fragments(mz=np.array([50.]), intensities=np.array([0.1]))
    metadata: dict
        Dict of metadata with for example the scan number of precursor m/z.

    """

    _peak_comments_mz_tolerance = 1e-05

    def __init__(self, mz: np.array,
                 intensities: np.array,
                 metadata: Optional[dict] = None,
                 harmonize_defaults: bool = False):
        """

        Parameters
        ----------
        mz
            Array of m/z for the peaks
        intensities
            Array of intensities for the peaks
        metadata
            Dictionary with for example the scan number of precursor m/z.
        harmonize_defaults : bool, optional
            Set to False if metadata harmonization to default keys is not desired.
            The default is True.  #TODO: temporarily set to False
        """
        self._metadata = Metadata(metadata, harmonize_defaults=harmonize_defaults)
        self.peaks = Fragments(mz=mz, intensities=intensities)
        self.losses = None


    def __eq__(self, other):
        return \
            self.peaks == other.peaks and \
            self.losses == other.losses and \
            self._metadata == other._metadata

    def __hash__(self):
        """Return a integer hash which is computed from both
        metadata (see .metadata_hash() method) and spectrum peaks
        (see .spectrum_hash() method)."""
        combined_hash = self.metadata_hash() + self.spectrum_hash()
        return int.from_bytes(bytearray(combined_hash, 'utf-8'), 'big')

    def spectrum_hash(self):
        """Return a (truncated) sha256-based hash which is generated
        based on the spectrum peaks (mz:intensity pairs).
        Spectra with same peaks will results in same spectrum_hash."""
        return spectrum_hash(self.peaks)

    def metadata_hash(self):
        """Return a (truncated) sha256-based hash which is generated
        based on the spectrum metadata.
        Spectra with same metadata results in same metadata_hash."""
        return metadata_hash(self._metadata.data)

    def clone(self):
        """Return a deepcopy of the spectrum instance."""
        clone = Spectrum(mz=self.peaks.mz,
                         intensities=self.peaks.intensities,
                         metadata=self._metadata.data)
        clone.losses = self.losses
        return clone

    def plot(self, intensity_from=0.0, intensity_to=None, with_histogram=False):
        """To visually inspect a spectrum run ``spectrum.plot()``

        .. figure:: ../_static/spectrum-plot-example.png
            :width: 400
            :alt: spectrum plotting function

            Example of a spectrum plotted using ``spectrum.plot()`` and ``spectrum.plot(intensity_to=0.02)``.."""

        def plot_histogram():
            """Plot the histogram of intensity values as horizontal bars, aligned with the spectrum axes"""

            def calc_bin_edges_intensity():
                """Calculate various properties of the histogram bins, given a range in intensity defined by
                'intensity_from' and 'intensity_to', assuming a number of bins equal to 100."""
                edges = np.linspace(intensity_from, intensity_to, n_bins + 1)
                lefts = edges[:-1]
                rights = edges[1:]
                middles = (lefts + rights) / 2
                widths = rights - lefts
                return edges, middles, widths

            bin_edges, bin_middles, bin_widths = calc_bin_edges_intensity()
            counts, _ = np.histogram(self.peaks.intensities, bins=bin_edges)
            histogram_ax.set_ylim(bottom=intensity_from, top=intensity_to)
            pyplot.barh(bin_middles, counts, height=bin_widths, color="#047495")
            pyplot.title(f"histogram (n_bins={n_bins})")
            pyplot.xlabel("count")

        def plot_spectrum():
            """plot mz v. intensity"""

            def make_stems():
                """calculate where the stems of the spectrum peaks are going to be"""
                x = np.zeros([2, self.peaks.mz.size], dtype="float")
                y = np.zeros(x.shape)
                x[:, :] = np.tile(self.peaks.mz, (2, 1))
                y[1, :] = self.peaks.intensities
                return x, y

            spectrum_ax.set_ylim(bottom=intensity_from, top=intensity_to)
            x, y = make_stems()
            pyplot.plot(x, y, color="#0f0f0f", linewidth=1.0, marker="")
            pyplot.title("Spectrum")
            pyplot.xlabel("M/z")
            pyplot.ylabel("intensity")

        if intensity_to is None:
            intensity_to = self.peaks.intensities.max() * 1.05

        n_bins = 100
        fig = pyplot.figure()

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

    def get(self, key: str, default=None):
        """Retrieve value from :attr:`metadata` dict. Shorthand for

        .. code-block:: python

            val = self.metadata[key]

        """
        return self._metadata.get(key, default)

    def set(self, key: str, value):
        """Set value in :attr:`metadata` dict. Shorthand for

        .. code-block:: python

            self.metadata[key] = val

        """
        self._metadata.set(key, value)
        return self

    @property
    def metadata(self):
        return self._metadata.data.copy()

    @metadata.setter
    def metadata(self, value):
        self._metadata.data = value

    @property
    def losses(self) -> Optional[Fragments]:
        return self._losses.clone() if self._losses is not None else None

    @losses.setter
    def losses(self, value: Fragments):
        self._losses = value

    @property
    def peaks(self) -> Fragments:
        return self._peaks.clone()

    @peaks.setter
    def peaks(self, value: Fragments):
        if isinstance(self.get("peak_comments"), dict):
            self._reiterate_peak_comments(value)
        self._peaks = value

    @property
    def peak_comments(self):
        return self.get("peak_comments")

    @peak_comments.setter
    def peak_comments(self, value):
        self.set("peak_comments", value)

    @classmethod
    def update_peak_comments_mz_tolerance(cls, mz_tolerance: float):
        cls._peak_comments_mz_tolerance = mz_tolerance

    def _reiterate_peak_comments(self, peaks: Fragments):
        """Update the peak comments to reflect the new peaks."""
        if not isinstance(self.get("peak_comments", None), dict):
            return None

        mz_tolerance = self._peak_comments_mz_tolerance

        def _append_new_comment(key):
            if new_key_comment is not None:
                comment = "; ".join([new_key_comment, self.metadata["peak_comments"].get(key)])
            else:
                comment = self.metadata["peak_comments"].get(key)
            return comment

        for key in list(self.metadata["peak_comments"].keys()):
            if key not in peaks.mz:
                if np.isclose(key, peaks.mz, rtol=mz_tolerance).any():
                    new_key = peaks.mz[np.isclose(key, peaks.mz, rtol=mz_tolerance).argmax()]
                    new_key_comment = self.metadata["peak_comments"].get(new_key, None)
                    new_key_comment = _append_new_comment(key)
                    self._metadata["peak_comments"][new_key] = new_key_comment
                self._metadata["peak_comments"].pop(key)
