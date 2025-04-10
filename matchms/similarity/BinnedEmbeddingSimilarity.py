from typing import Iterable
import numpy as np
from matchms.typing import SpectrumType
from .BaseEmbeddingSimilarity import BaseEmbeddingSimilarity


class BinnedEmbeddingSimilarity(BaseEmbeddingSimilarity):
    """A similarity measure that bins spectra into a fixed number of bins and uses the binned
    intensities as embedding features. By default, the similarity between spectra is computed as the cosine
    similarity between their binned representations.

    Parameters
    ----------
    similarity : str, optional
        The similarity measure to use for comparing embeddings. Default is "cosine".
        Options are "cosine" or "euclidean".
    max_mz : float, optional
        The maximum m/z value to consider when binning. Default is 1005.
    bin_width : float, optional
        The width of each bin in m/z units. Default is 1.
    """
    def __init__(self, similarity: str = "cosine", max_mz: float = 1005, bin_width: float = 1):
        super().__init__(similarity=similarity)
        self.max_mz = max_mz
        self.bin_width = bin_width

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
        # NOTE: copypaste from https://github.com/pluskal-lab/MassSpecGym/blob/f525a5e55a39ec4caa4f1a51e64acd046713179e/massspecgym/data/transforms.py#L97
        mzs = spectrum.peaks.mz
        intensities = spectrum.peaks.intensities

        # Calculate the number of bins
        num_bins = int(np.ceil(self.max_mz / self.bin_width))

        # Calculate the bin indices for each mass
        bin_indices = np.floor(mzs / self.bin_width).astype(int)

        # Filter out mzs that exceed max_mz
        valid_indices = bin_indices[mzs <= self.max_mz]
        valid_intensities = intensities[mzs <= self.max_mz]

        # Clip bin indices to ensure they are within the valid range
        valid_indices = np.clip(valid_indices, 0, num_bins - 1)

        # Initialize an array to store the binned intensities
        binned_intensities = np.zeros(num_bins)

        # Use np.add.at to sum intensities in the appropriate bins
        np.add.at(binned_intensities, valid_indices, valid_intensities)

        # Normalize the intensities to relative intensities
        binned_intensities /= np.max(binned_intensities)

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
        spectra_list = list(spectra)
        embeddings = []

        for spectrum in spectra_list:
            binned_spectrum = self._bin_spectrum(spectrum)
            embeddings.append(binned_spectrum)

        return np.array(embeddings)
