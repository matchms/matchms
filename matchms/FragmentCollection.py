from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from functools import cached_property
import numpy as np
from scipy.sparse import coo_array, csr_array
from tqdm.auto import tqdm
from matchms.Spectrum import Spectrum
from .hashing import spectra_hashes
from .typing import FragmentCollectionType


class FragmentCollection(ABC):
    """Abstract base class for a collection of spectra fragments."""
    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """Return (n_spectra, n_bins)."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def copy(self) -> 'FragmentCollection':
        pass

    @abstractmethod
    def get_row(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (mz, intensities) for a single row."""
        pass

    @abstractmethod
    def take(self, indices: Iterable[int]) -> 'FragmentCollection':
        """Return new collection with selected rows."""
        pass

    @abstractmethod
    def slice_mz(self, mz_min: float | None = None, mz_max: float | None = None) -> 'FragmentCollection':
        """Return new collection with restricted m/z range."""
        pass

    @abstractmethod
    def sum(self, axis: int = 1) -> np.ndarray:
        pass

    @abstractmethod
    def count(self, axis: int = 1) -> np.ndarray:
        pass

    @abstractmethod
    def mz_to_bin(self, mz: np.ndarray | float) -> np.ndarray:
        pass

    @abstractmethod
    def bin_to_mz(self, bin_idx: np.ndarray | int) -> np.ndarray:
        pass

    @abstractmethod
    def count_peaks_above_relative_intensity(
        self,
        intensity_from: float,
    ) -> np.ndarray:
        """Return number of peaks per row with relative intensity >= intensity_from."""
        pass

    # Filtering methods for peak processing filters.
    @abstractmethod
    def select_by_intensity(
        self,
        intensity_from: float = 0.0,
        intensity_to: float = 1.0,
    ) -> "FragmentCollection":
        """Return new collection with peaks restricted to an intensity range."""
        pass

    @abstractmethod
    def select_by_relative_intensity(
        self,
        intensity_from: float = 0.0,
        intensity_to: float = 1.0,
    ) -> "FragmentCollection":
        """Return new collection with peaks restricted to a row-wise relative intensity range."""
        pass

    @abstractmethod
    def keep_top_k_per_row_variable(self, k_per_row: np.ndarray) -> 'FragmentCollection':
        """Return new collection with only the top-k intensity peaks per row."""
        pass


class CSRFragmentCollection(FragmentCollection):
    """CSR-backed, binned fragment storage for a spectra dataset.

    Stores all fragments of a dataset in a sparse matrix using CSR format:

    - rows correspond to spectra
    - columns correspond to discrete m/z bins
    - values correspond to peak intensities

    The m/z values of input peaks are converted to integer bin indices using
    :meth:`mz_to_bin`. This means that the original m/z values are not stored
    directly. When spectra are reconstructed, m/z values are returned as bin
    centers via :meth:`bin_to_mz`.

    Notes
    -----
    This is a binned representation and is therefore not necessarily lossless.
    If multiple peaks from the same spectrum fall into the same m/z bin, they are
    stored at the same sparse matrix coordinate. During sparse matrix construction,
    such duplicate coordinates are combined by summing their intensities.

    For example, two peaks in one spectrum that map to the same m/z bin will be
    represented as a single peak with the summed intensity when the row is
    reconstructed.

    The choice of ``bin_size`` therefore controls both mass precision and the
    likelihood of peak merging. Smaller bin sizes preserve m/z differences more
    closely but may create very large sparse matrix dimensions. Larger bin sizes
    reduce the number of bins but increase the chance that neighboring peaks are
    merged.
    """

    def __init__(
        self,
        spectra: list[Spectrum] | Generator[Spectrum, None, None] | None = None,
        *,
        array: csr_array | None = None,
        bin_size: float = 1e-6,
        index_dtype: np.dtype = np.int64,
    ):
        if bin_size <= 0:
            raise ValueError("bin_size must be > 0.")
        self.bin_size = float(bin_size)
        self.index_dtype = index_dtype

        if array is not None:
            if spectra is not None:
                raise ValueError("Pass either spectra or array, not both.")
            self._array = array.tocsr()
            return

        if spectra is None:
            raise ValueError("Either spectra or array must be provided.")

        spectra = list(spectra)
        if not spectra:
            raise ValueError("Spectra must contain at least one Spectrum.")

        self._array = self._construct_from_spectra_list(spectra)

    @classmethod
    def from_array(cls, array: csr_array, *, bin_size: float = 1e-6) -> FragmentCollectionType:
        return cls(array=array, bin_size=bin_size)

    def _construct_from_spectra_list(self, spectra: list[Spectrum]) -> csr_array:
        lengths = np.array([len(spec.mz) for spec in spectra])

        if lengths.sum() == 0:
            return csr_array((len(spectra), 0), dtype=np.float32)

        all_mz = np.concatenate([spec.mz for spec in spectra])
        all_int = np.concatenate([spec.intensities for spec in spectra])

        bin_idx = self.mz_to_bin(all_mz)
        n_bins = int(bin_idx.max()) + 1

        row_idx = np.repeat(np.arange(len(spectra), dtype=self.index_dtype), lengths)

        return coo_array(
            (all_int, (row_idx, bin_idx)),
            shape=(len(spectra), n_bins),
        ).tocsr()

    @property
    def array(self) -> csr_array:
        return self._array

    @property
    def shape(self) -> tuple[int, int]:
        return self._array.shape

    @property
    def n_spectra(self) -> int:
        return self._array.shape[0]

    @property
    def n_bins(self) -> int:
        return self._array.shape[1]

    def __len__(self) -> int:
        return self.n_spectra

    def __repr__(self) -> str:
        return (
            f"CSRFragmentCollection(n_spectra={self.n_spectra}, "
            f"n_fragments={self._array.data.shape[0]}, bin_size={self.bin_size})"
        )

    def copy(self) -> FragmentCollectionType:
        return self.__class__.from_array(self._array.copy(), bin_size=self.bin_size)

    def mz_to_bin(self, mz: np.ndarray | float) -> np.ndarray:
        """Convert m/z values to integer bins."""
        return np.floor(np.asarray(mz) / self.bin_size).astype(self.index_dtype)

    def bin_to_mz(self, bin_idx: np.ndarray | int) -> np.ndarray:
        """Convert bin indices to bin-center m/z values."""
        return (bin_idx * self.bin_size) + (self.bin_size / 2)

    def get_row(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one spectrum row as (mz, intensities)."""
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("row index out of range")

        csr = self._array
        start, end = csr.indptr[idx], csr.indptr[idx + 1]
        cols = csr.indices[start:end]
        intensities = csr.data[start:end]
        mz = self.bin_to_mz(cols)
        return mz, intensities.copy()

    def iter_peak_arrays(self):
        """Yield rows as `(mz, intensities)` tuples."""
        for i in range(len(self)):
            yield self.get_row(i)

    def to_peak_arrays(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return all rows as a list of `(mz, intensities)` tuples."""
        return list(self.iter_peak_arrays())

    def take(self, indices: Iterable[int]) -> FragmentCollectionType:
        """Return a new collection with selected rows in the given order."""
        indices = np.asarray(list(indices), dtype=self.index_dtype)
        return self.__class__.from_array(self._array[indices, :], bin_size=self.bin_size)

    def reorder(self, indices: Iterable[int]) -> FragmentCollectionType:
        """Alias for take()."""
        return self.take(indices)

    def filter(self, mask: np.ndarray | list[bool]) -> FragmentCollectionType:
        """Return a new collection keeping rows where mask is True."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != len(self):
            raise ValueError(
                f"Mask length ({mask.shape[0]}) does not match number of spectra ({len(self)})."
            )
        return self.take(np.where(mask)[0])

    def drop(self, indices: Iterable[int]) -> FragmentCollectionType:
        """Return a new collection with selected rows removed."""
        indices = np.asarray(list(indices), dtype=self.index_dtype)
        all_indices = np.arange(len(self))
        keep_mask = ~np.isin(all_indices, indices)
        return self.take(all_indices[keep_mask])

    def drop_empty(self) -> FragmentCollectionType:
        """Return a new collection without rows that have no peaks."""
        return self.filter(self.count(axis=1) > 0)

    def slice_rows(self, rows) -> FragmentCollectionType:
        """Return a row-sliced collection."""
        if isinstance(rows, slice):
            indices = np.arange(len(self))[rows]
            return self.take(indices)
        if isinstance(rows, (list, np.ndarray)):
            arr = np.asarray(rows)
            if arr.dtype == bool:
                return self.filter(arr)
            return self.take(arr)
        if isinstance(rows, (int, np.integer)):
            return self.take([int(rows)])
        raise TypeError("Unsupported row selector.")

    def slice_mz(self, mz_min: float | None = None, mz_max: float | None = None):
        """Return a new collection restricted to an m/z window.

        Notes
        -----
        This keeps the global bin coordinate system unchanged.
        Bins outside the requested m/z range are removed from the data, but the
        matrix shape and column numbering remain unchanged.
        """
        start_bin = 0 if mz_min is None else int(self.mz_to_bin(mz_min))
        stop_bin = self.n_bins if mz_max is None else int(self.mz_to_bin(mz_max)) + 1

        start_bin = max(0, min(self.n_bins, start_bin))
        stop_bin = max(0, min(self.n_bins, stop_bin))

        if mz_min is not None and mz_max is not None and mz_max < mz_min:
            raise ValueError("mz_max must be >= mz_min.")

        coo = self._array.tocoo()
        keep = (coo.col >= start_bin) & (coo.col < stop_bin)

        new_array = coo_array(
            (coo.data[keep], (coo.row[keep], coo.col[keep])),
            shape=self._array.shape,
        ).tocsr()

        return self.__class__.from_array(new_array, bin_size=self.bin_size)

    def __getitem__(self, key):
        """Support row slicing and optional row/column slicing.

        Examples
        --------
        rows only:
            fragments[:500]
            fragments[[1, 4, 8]]
            fragments[mask]

        rows + m/z float slicing:
            fragments[:500, 100.0:205.5]
        """
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Expected at most two indexers: rows, mz-range")
            row_sel, col_sel = key
            row_sliced = self.slice_rows(row_sel)

            if isinstance(col_sel, slice):
                if (
                    (col_sel.start is None or isinstance(col_sel.start, (float, int, np.floating, np.integer)))
                    and (col_sel.stop is None or isinstance(col_sel.stop, (float, int, np.floating, np.integer)))
                    and (col_sel.step is None)
                ):
                    # Interpret m/z axis slicing always as m/z value --> convert to bins
                    return row_sliced.slice_mz(col_sel.start, col_sel.stop)
            raise TypeError("Unsupported column selector for CSRFragmentCollection.")

        return self.slice_rows(key)

    def sum(self, axis: int = 1, **kwargs):
        result = self._array.sum(axis=axis, **kwargs)
        if axis == 1:
            return result.A1 if hasattr(result, "A1") else np.asarray(result).ravel()
        return result

    def max(self, axis: int = 1, **kwargs):
        return self._array.max(axis=axis, **kwargs)

    def min(self, axis: int = 1, **kwargs):
        return self._array.min(axis=axis, **kwargs)

    def mean(self, axis: int = 1, **kwargs):
        return self._array.mean(axis=axis, **kwargs)

    def count(self, axis: int = 1):
        """Count nonzero peaks per row or per bin."""
        if axis == 1:
            return np.diff(self._array.indptr)
        if axis == 0:
            return np.bincount(self._array.indices, minlength=self._array.shape[1])
        raise ValueError("axis must be 0 or 1")

    def count_peaks_above_relative_intensity(
            self,
            intensity_from: float,
        ) -> np.ndarray:
        """Return number of peaks per row with relative intensity >= intensity_from."""
        if intensity_from < 0.0:
            raise ValueError("'intensity_from' should be larger than or equal to 0.")
        if intensity_from > 1.0:
            raise ValueError("'intensity_from' should be smaller than or equal to 1.0.")

        coo = self._array.tocoo()
        counts = np.zeros(len(self), dtype=np.int64)

        if coo.data.size == 0:
            return counts

        row_max = self._array.max(axis=1)
        if hasattr(row_max, "toarray"):
            row_max = row_max.toarray()
        row_max = np.asarray(row_max).ravel()

        valid = row_max[coo.row] > 0
        relative_intensities = np.zeros_like(coo.data, dtype=float)
        relative_intensities[valid] = coo.data[valid] / row_max[coo.row[valid]]

        keep = valid & (relative_intensities >= intensity_from)

        return np.bincount(coo.row[keep], minlength=len(self))

    def row_intensity_sums(self) -> np.ndarray:
        return self.sum(axis=1)

    def row_peak_counts(self) -> np.ndarray:
        return self.count(axis=1)

    @cached_property
    def fragment_hashes(self):
        return spectra_hashes(self._array, self.bin_to_mz)

    # --------------------------------------------
    # Abstract methods for peak processing filters
    # --------------------------------------------
    def select_by_intensity(
            self,
            intensity_from: float = 0.0,
            intensity_to: float = 1.0,
        ) -> FragmentCollectionType:
        """Return a new collection keeping peaks within an intensity range."""
        if intensity_from > intensity_to:
            raise ValueError(
                "'intensity_from' should be smaller than or equal to 'intensity_to'."
            )

        coo = self._array.tocoo()
        keep = (coo.data >= intensity_from) & (coo.data <= intensity_to)

        new_array = coo_array(
            (coo.data[keep], (coo.row[keep], coo.col[keep])),
            shape=self._array.shape,
        ).tocsr()

        return self.__class__.from_array(new_array, bin_size=self.bin_size)

    def select_by_relative_intensity(
            self,
            intensity_from: float = 0.0,
            intensity_to: float = 1.0,
        ) -> FragmentCollectionType:
        """Return a new collection keeping peaks within a row-wise relative intensity range."""
        if intensity_from < 0.0:
            raise ValueError("'intensity_from' should be larger than or equal to 0.")
        if intensity_to > 1.0:
            raise ValueError("'intensity_to' should be smaller than or equal to 1.0.")
        if intensity_from > intensity_to:
            raise ValueError(
                "'intensity_from' should be smaller than or equal to 'intensity_to'."
            )

        coo = self._array.tocoo()

        if coo.data.size == 0:
            return self.__class__.from_array(self._array.copy(), bin_size=self.bin_size)

        row_max = self._array.max(axis=1).toarray().ravel()

        nonzero_row_max = row_max[coo.row] > 0

        relative_intensities = np.zeros_like(coo.data, dtype=float)
        relative_intensities[nonzero_row_max] = (
            coo.data[nonzero_row_max] / row_max[coo.row[nonzero_row_max]]
        )

        keep = (
            nonzero_row_max
            & (relative_intensities >= intensity_from)
            & (relative_intensities <= intensity_to)
        )

        new_array = coo_array(
            (coo.data[keep], (coo.row[keep], coo.col[keep])),
            shape=self._array.shape,
        ).tocsr()

        return self.__class__.from_array(new_array, bin_size=self.bin_size)

    def keep_top_k_per_row_variable(
            self,
            k_per_row: np.ndarray,
            progress_bar: bool = False,
            ) -> FragmentCollectionType:
        """Keep the top-k highest-intensity peaks per row.

        Parameters
        ----------
        k_per_row:
            One integer value per spectrum row. For each row, only the k highest
            intensity peaks are retained. Remaining peaks are sorted by m/z/bin
            position, preserving normal sparse row order.
        progress_bar:
            Whether to display a progress bar when processing large collections.
        """
        k_per_row = np.asarray(k_per_row)

        if k_per_row.shape[0] != len(self):
            raise ValueError(
                f"k_per_row length ({k_per_row.shape[0]}) does not match "
                f"number of spectra ({len(self)})."
            )

        if np.any(k_per_row < 0):
            raise ValueError("k_per_row values must be non-negative.")

        csr = self._array

        data_parts = []
        index_parts = []
        indptr = [0]

        for row_idx in tqdm(range(len(self)), disable=not progress_bar):
            start, end = csr.indptr[row_idx], csr.indptr[row_idx + 1]
            row_data = csr.data[start:end]
            row_indices = csr.indices[start:end]

            n_peaks = row_data.size
            k = int(k_per_row[row_idx])

            if k >= n_peaks:
                keep = np.arange(n_peaks)
            elif k == 0:
                keep = np.array([], dtype=np.int64)
            else:
                # Select k largest intensities without fully sorting the row.
                keep = np.argpartition(row_data, -k)[-k:]

                # Restore m/z/bin order, matching Spectrum implementation behavior.
                keep = keep[np.argsort(row_indices[keep])]

            data_parts.append(row_data[keep])
            index_parts.append(row_indices[keep])
            indptr.append(indptr[-1] + keep.size)

        if len(data_parts) > 0:
            data = np.concatenate(data_parts).astype(csr.data.dtype, copy=False)
            indices = np.concatenate(index_parts).astype(csr.indices.dtype, copy=False)
        else:
            data = np.array([], dtype=csr.data.dtype)
            indices = np.array([], dtype=csr.indices.dtype)

        new_array = csr_array(
            (data, indices, np.asarray(indptr, dtype=csr.indptr.dtype)),
            shape=csr.shape,
        )

        return self.__class__.from_array(new_array, bin_size=self.bin_size)
