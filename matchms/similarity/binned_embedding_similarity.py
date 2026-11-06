from collections.abc import Iterable
import numpy as np
from matchms.typing import SpectrumType
from .base_embedding_similarity import BaseEmbeddingSimilarity


class BinnedEmbeddingSimilarity(BaseEmbeddingSimilarity):
    """Compare spectra by cosine/euclidean similarity of binned intensities.

    Spectra are converted to fixed-length vectors by summing intensities in
    equally spaced m/z bins. Each vector is normalized to its maximum bin
    intensity when that maximum is positive. Empty spectra, spectra without peaks
    in the configured m/z range, and spectra with only zero intensities produce a
    zero vector instead of NaNs.

    Parameters
    ----------
    similarity
        Similarity measure used for comparing embeddings. Supported values are
        ``"cosine"`` and ``"euclidean"``.
    max_mz
        Maximum m/z value to include. Values outside ``[0, max_mz]`` are ignored.
    bin_width
        Width of each m/z bin.
    intensity_power
        Power applied to peak intensities before binning.
    """
    def __init__(
        self, similarity: str = "cosine",
        max_mz: float = 1005,
        bin_width: float = 1,
        intensity_power: float = 1
    ):
        super().__init__(similarity=similarity)
        if max_mz <= 0:
            raise ValueError("max_mz must be > 0.")
        if bin_width <= 0:
            raise ValueError("bin_width must be > 0.")

        self.max_mz = max_mz
        self.bin_width = bin_width
        self.intensity_power = intensity_power

    @property
    def n_bins(self) -> int:
        """Number of bins used for each embedding vector."""
        return int(np.floor(self.max_mz / self.bin_width)) + 1

    def _bin_spectrum(self, spectrum: SpectrumType) -> np.ndarray:
        """Bin a spectrum's peaks into fixed-width m/z bins.

        Parameters
        ----------
        spectrum : SpectrumType
            The spectrum to bin.

        Returns
        -------
        np.ndarray
            Array of binned and normalized intensities.
        """
        mzs = spectrum.peaks.mz
        intensities = spectrum.peaks.intensities
        binned_intensities = np.zeros(self.n_bins, dtype=np.float64)

        if mzs.size == 0:
            return binned_intensities

        valid_mask = (mzs >= 0) & (mzs <= self.max_mz)
        if not np.any(valid_mask):
            return binned_intensities

        valid_mzs = mzs[valid_mask]
        valid_intensities = intensities[valid_mask] ** self.intensity_power

        bin_indices = np.floor(valid_mzs / self.bin_width).astype(np.int64)
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        np.add.at(binned_intensities, bin_indices, valid_intensities)

        max_intensity = np.max(binned_intensities)
        if max_intensity > 0:
            binned_intensities /= max_intensity

        return binned_intensities

    def compute_embeddings(self, spectra: Iterable[SpectrumType]) -> np.ndarray:
        """Convert spectra into binned embeddings.

        Parameters
        ----------
        spectra : Iterable[SpectrumType]
            The spectra to convert into embeddings.

        Returns
        -------
        np.ndarray
            Array of shape (n_spectra, n_bins) containing the binned embeddings.
        """
        embeddings = [self._bin_spectrum(spectrum) for spectrum in spectra]
        if not embeddings:
            return np.zeros((0, self.n_bins), dtype=np.float64)
        return np.vstack(embeddings)
