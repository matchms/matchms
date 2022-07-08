from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from matchms.plotting.spectrum_plots import plot_spectra_mirror, plot_spectrum
from .filtering.add_precursor_mz import _add_precursor_mz_metadata
from .filtering.add_retention import _add_retention
from .filtering.interpret_pepmass import _interpret_pepmass_metadata
from .filtering.make_charge_int import _convert_charge_to_int
from .Fragments import Fragments
from .hashing import metadata_hash, spectrum_hash
from .Metadata import Metadata


class Spectrum:
    """Container for a collection of peaks, losses and metadata.

    Spectrum peaks are stored as :class:`~matchms.Fragments` object which can be
    addressed calling `spectrum.peaks` and contains m/z values and the respective
    peak intensities.

    Spectrum metadata is stored as :class:`~matchms.Metadata` object which can be
    addressed by `spectrum.metadata`.

    Code example

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
                 metadata_harmonization: bool = True):
        """

        Parameters
        ----------
        mz
            Array of m/z for the peaks
        intensities
            Array of intensities for the peaks
        metadata
            Dictionary with for example the scan number of precursor m/z.
        metadata_harmonization : bool, optional
            Set to False if default metadata filters should not be applied.
            The default is True.
        """
        self._metadata = Metadata(metadata)
        if metadata_harmonization is True:
            self._apply_metadata_harmonization()
        self.peaks = Fragments(mz=mz, intensities=intensities)
        self.losses = None

    def __eq__(self, other):
        return \
            self.peaks == other.peaks and \
            self.losses == other.losses and \
            self._metadata == other._metadata

    def _apply_metadata_harmonization(self):
        metadata_filtered = _interpret_pepmass_metadata(self.metadata)
        if metadata_filtered.get("ionmode") is not None:
            metadata_filtered["ionmode"] = self.metadata.get("ionmode").lower()
        metadata_filtered = _add_precursor_mz_metadata(metadata_filtered)

        if metadata_filtered.get("retention_time") is not None:
            metadata_filtered = _add_retention(metadata_filtered, "retention_time", "retention_time")
        if metadata_filtered.get("retention_index") is not None:
            metadata_filtered = _add_retention(metadata_filtered, "retention_index", "retention_index")
        charge = metadata_filtered.get("charge")
        if not isinstance(charge, int) and not _convert_charge_to_int(charge) is None:
            metadata_filtered["charge"] = _convert_charge_to_int(charge)
        self._metadata = Metadata(metadata_filtered)

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
                         metadata=self._metadata.data,
                         metadata_harmonization=False)
        clone.losses = self.losses
        return clone

    def plot(self, figsize=(8, 6), dpi=200, **kwargs):
        """Plot to visually inspect a spectrum run ``spectrum.plot()``

        .. figure:: ../_static/spectrum-plot-example.png
            :width: 450
            :alt: spectrum plotting function

            Example of a spectrum plotted using ``spectrum.plot()`` ..
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax = plot_spectrum(self, ax=ax, **kwargs)
        return fig, ax

    def plot_against(self, other_spectrum,
                     figsize=(8, 6), dpi=200,
                     **spectrum_kws):
        """Compare two spectra visually in a mirror plot.

        To visually compare the peaks of two spectra run
        ``spectrum.plot_against(other_spectrum)``

        .. figure:: ../_static/spectrum-mirror-plot-example.png
            :width: 450
            :alt: spectrum mirror plot function

            Example of a mirror plot comparing two spectra ``spectrum.plot_against()`` ..
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax = plot_spectra_mirror(self, other_spectrum, ax=ax, **spectrum_kws)
        return fig, ax

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

    def to_dict(self) -> dict:
        """Return a dictionary representation of a spectrum."""
        peaks_list = np.vstack((self.peaks.mz, self.peaks.intensities)).T.tolist()
        spectrum_dict = dict(self.metadata.items())
        spectrum_dict["peaks_json"] = peaks_list
        if "fingerprint" in spectrum_dict:
            spectrum_dict["fingerprint"] = spectrum_dict["fingerprint"].tolist()
        return spectrum_dict

    @property
    def mz(self):
        return self.peaks.mz

    @property
    def intensities(self):
        return self.peaks.intensities

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
