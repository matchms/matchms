from __future__ import annotations
from functools import cached_property
from typing import Generator
import numpy as np
import pandas as pd
from scipy.sparse import coo_array
from matchms.Spectrum import Spectrum
from .hashing import compute_combined_hashes, spectra_hashes


class SpectraCollection:
    def __init__(self, spectra: list[Spectrum] | Generator[Spectrum, None, None], bin_size=0.000001):
        spectra = list(spectra)

        if not spectra:
            raise ValueError("Spectra must contain at least one Spectrum.")

        self.bin_size = bin_size
        self._metadata = self._construct_metadata(spectra)
        self._fragments = self._construct_fragments(spectra)

        if len(self._metadata) != self._fragments.shape[0]:
            raise ValueError("Spectra Metadata/Fragments mismatch.")

    def _construct_fragments(self, spectra: list):
        all_mz = np.concatenate([spec.mz for spec in spectra])
        all_int = np.concatenate([spec.intensities for spec in spectra])

        bin_idx = self.mz_to_bin(all_mz)
        bin_no = bin_idx.max() + 1

        lengths = np.array([len(spec.mz) for spec in spectra])
        row_idx = np.repeat(np.arange(len(spectra)), lengths)

        return coo_array((all_int, (row_idx, bin_idx)), shape=(len(spectra), bin_no)).tocsr()  # CSR here!

    def _construct_metadata(self, spectra):
        # data = defaultdict(list)
        # [data[k].append(v) for spectrum in spectra for k, v in spectrum.metadata.items()]
        # TODO: add minimal Matadata harmonization

        # return pd.DataFrame(data)
        records = [spectrum.metadata for spectrum in spectra]
        return pd.DataFrame.from_records(records)

    @property
    def metadata(self) -> pd.DataFrame:
        return MetadataProxy(self._metadata, self)

    @property
    def fragments(self) -> FragmentsProxy:
        return FragmentsProxy(self._fragments)

    @property
    def shape(self):
        return self._fragments.shape[0], self._metadata.shape[1]

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            csr = self._fragments

            start, end = csr.indptr[idx], csr.indptr[idx + 1]
            cols = csr.indices[start:end]
            intensities = csr.data[start:end]
            mz = self.bin_to_mz(cols)

            return Spectrum(mz=mz, intensities=intensities, metadata=self._metadata.iloc[idx].to_dict())

        target = self.copy()
        if isinstance(idx, slice):
            indices = np.arange(len(self))[idx]
        else:
            indices = idx

        return target._reorder(indices)

    def __iter__(self):
        for i in range(self._fragments.shape[0]):
            yield self[i]

    def __len__(self):
        return self._fragments.shape[0]

    def __repr__(self):
        rep = f"Spectra in list: {len(self._metadata)}"

        return rep

    def __str__(self):
        return self.__repr__()

    @property
    def spectra_hashes(self):
        return compute_combined_hashes(self.fragment_hashes, self.metadata_hashes)

    @cached_property
    def fragment_hashes(self):
        return spectra_hashes(self.fragments._array, self.bin_to_mz)

    @cached_property
    def metadata_hashes(self):
        return pd.util.hash_pandas_object(self._metadata, index=False).tolist()

    def add_metadata(self, data, col_name: str = None, overwrite: bool = False):
        # TODO: must contain same sorting as present spectra/metadata. Add class bool flag, if data has been sorted?
        if isinstance(data, pd.DataFrame):
            new_metadata = data.copy()

        if isinstance(data, pd.Series):
            col_name = col_name or data.name
            if col_name is None:
                raise ValueError("Series must have a name or 'col_name' must be provided.")
            new_metadata = pd.DataFrame({col_name: data})

        if isinstance(data, list):
            if col_name is None:
                raise ValueError("'col_name' must be provided.")
            new_metadata = pd.DataFrame({col_name: data})

        if isinstance(data, dict):
            for v in data.values():
                if not isinstance(v, list):
                    raise ValueError("When data is a dict, values must be of type list.")

            new_metadata = pd.DataFrame(data)

        if new_metadata.shape[0] != len(self):
            raise ValueError("New metadata does not match length of existing metadata entries.")

        overlap = self._metadata.columns.intersection(new_metadata.columns)
        if not overlap.empty:
            if not overwrite:
                raise ValueError(f"Columns already exist: {list(overlap)}. Set overwrite to True to replace values.")

            self._metadata = self._metadata.drop(columns=overlap)

        self._metadata = pd.concat([self._metadata, new_metadata], axis=1)
        self._clear_cache(["metadata_hashes"])

    def _reorder(self, indices: np.ndarray):
        """
        Reorders fragments and metadata in SpectrumCollection according to indices synchronically.

        Parameters:
        -----------
        inplace : bool
            Will return a new SpectraCollection, if True and the same if False. Defaults to False.
        """
        self._fragments = self._fragments[indices, :]
        self._metadata = self._metadata.iloc[indices].reset_index(drop=True)

        return self

    def sort(self, by: str | list[str], on: str = "metadata", inplace: bool = False, **kwargs):
        """
        Sorts SpectraCollection (fragments AND metadata) by either metadata keyword(s) or fragment function.

        Parameters:
        -----------
        by : str | list[str]
            Either metadata column name or method name in FragmentsProxy (e.g., 'sum').
        on : str
            'metadata' (Standard) or 'fragments'.
        inplace : bool
            Will return a new, sorted SpectraCollection, if True and the same, sorted if False. Defaults to False.
        """
        target = self if inplace else self.copy()
        ascending = kwargs.get("ascending", True)

        if on == "fragments":
            if not hasattr(target.fragments, str(by)):
                raise NotImplementedError(f"'Sorting method {by}' is not implemented in FragmentsProxy.")

            method = getattr(target.fragments, str(by))
            sort_values = method(axis=1)

            new_indices = np.argsort(sort_values)
            if not ascending:
                new_indices = new_indices[::-1]

        elif on == "metadata":
            sorted_df = target._metadata.sort_values(by=by, **kwargs)
            new_indices = sorted_df.index.values

        else:
            raise ValueError("Parameter 'on' must be either 'metadata' or 'fragments'.")

        target._reorder(new_indices)

        return None if inplace else target

    def filter(self, mask: np.ndarray | pd.Series | list[bool], inplace: bool = False):
        """
        Filters SpectraCollection by keeping only the spectra where the mask is True.

        This method synchronizes the filtering of both fragments and metadata.
        It uses boolean indexing from NumPy and Pandas.

        Parameters
        ----------
            mask (np.ndarray | pd.Series | list[bool]): A boolean array-like object
                of the same length as the collection. Rows where the mask is True
                will be kept; all others will be removed.
            inplace (bool): If True, modifies the current collection in place and
                returns None. If False (default), returns a new filtered
                SpectraCollection instance.

        Returns
        -------
            SpectraCollection | None: A new filtered instance if inplace is False,
                otherwise None.

        Raises
            ValueError: If the length of the mask does not match the number of spectra in the collection.

        Example:
            >>> # Filter by metadata
            >>> filtered_coll = coll.filter(coll.metadata["ms_level"] == 2)
            >>>
            >>> # Filter by fragment properties
            >>> coll.filter(coll.fragments.sum() > 500, inplace=True)
            >>>
            >>> # Using an external vectorized filter function
            >>> mask = filter_min_peaks(coll, n_required=10)
            >>> coll.filter(mask, inplace=True)
        """
        if isinstance(mask, pd.Series):
            mask = mask.values
        mask = np.asanyarray(mask, dtype=bool)

        if mask.shape[0] != len(self):
            raise ValueError(f"Shape of filter mask ({mask.shape[0]}) does not fit Items in SpectraCollection ({len(self)}).")

        target = self if inplace else self.copy()

        keep_indices = np.where(mask)[0]
        target._reorder(keep_indices)

        return None if inplace else target

    def _clear_cache(self, keys: list[str] = None):
        if keys is None:
            keys = ["metadata_hashes", "fragment_hashes", "spectra_hashes"]

        for key in keys:
            self.__dict__.pop(key, None)

    def drop(self, indices: list[int] | np.ndarray, inplace: bool = False):
        """
        Removes specified rows (spectra) from both fragments and metadata.

        Parameters:
        -----------
        indices : list[int] | np.ndarray
            Indices of the rows to remove.
        inplace : bool
            Will return a new SpectraCollection, if True and the same if False. Defaults to False.
        """
        target = self if inplace else self.copy()

        all_indices = np.arange(target._fragments.shape[0])
        keep_mask = ~np.isin(all_indices, indices)

        target._fragments = target._fragments[keep_mask, :]
        target._metadata = target._metadata.iloc[keep_mask].reset_index(drop=True)

        target._clear_cache()

        return None if inplace else target

    def dropna(self, inplace: bool = False):
        """
        Removes spectra without peaks.

        Parameters:
        -----------
        inplace : bool
            Will return a new SpectraCollection, if True and the same if False. Defaults to False.
        """
        peaks_per_row = np.diff(self._fragments.indptr)
        empty_indices = np.where(peaks_per_row == 0)[0]

        if len(empty_indices) > 0:
            return self.drop(empty_indices, inplace=inplace)

        return self if inplace else self.copy()

    def drop_duplicates(self, inplace: bool = False):
        """
        Drops duplicates by spectra hashes.

        Parameters:
        -----------
        inplace : bool
            Will return a new SpectraCollection, if True and the same if False. Defaults to False.
        """
        _, unique_indices = np.unique(self.spectra_hashes, return_index=True)

        all_indices = np.arange(len(self.spectra_hashes))
        duplicate_indices = np.setdiff1d(all_indices, unique_indices)

        return self.drop(duplicate_indices, inplace=inplace)

    def copy(self):
        new_spec = self.__class__.__new__(self.__class__)
        new_spec.bin_size = self.bin_size
        new_spec._metadata = self._metadata.copy()
        new_spec._fragments = self._fragments.copy()

        return new_spec

    def mz_to_bin(self, mz: np.ndarray | float) -> np.ndarray:
        """
        Convert mz values into bins.

        Uses the bin_size of SpectraCollection and maps mz values into integer bins by flooring them.

        Parameters
        ----------
        mz
            The mz values to bin.

        Returns
        -------
        np.ndarray
            Bin indices as np.int64.

        TODO
        ----
        uint64 can lead to conversion issues with scipy sparse -> int64 sufficient?
        """
        return np.floor(mz / self.bin_size).astype(np.int64)

    def bin_to_mz(self, bin_idx: np.ndarray | int) -> np.ndarray:
        """
        Convert bin indices to mz values.

        Uses the bin_size of SpectraCollection and calculates the mz value at the center of the bin.

        Parameters
        ----------
        bin_idx
            Bin indices/columns to convert.

        Returns
        -------
        np.ndarray
            The mz values at the center of specified bins.
        """
        return (bin_idx * self.bin_size) + (self.bin_size / 2)

    def describe(self) -> pd.DataFrame:
        """
        Generate descriptive statistics for the spectra collection.

        Calculates key metrics for spectra collection,
        including peak counts, total ion intensity, average m/z, and Shannon
        entropy based on peak intensities. It then computes summary statistics
        (count, mean, std, min, max, etc.) for the entire collection.

        Returns:
            pd.DataFrame: A DataFrame containing summary statistics for the
                following columns:
                - 'peak_counts': Number of detected peaks per spectrum.
                - 'intensity_sums': Total ion current (TIC) per spectrum.
                - 'intensity_entropy': Shannon entropy of peak intensities,
                    quantifying the spectral complexity/information density.
        """
        peak_counts = np.diff(self._fragments.indptr)
        intensity_sums = np.asarray(self._fragments.sum(axis=1)).flatten()

        entropies = np.zeros(len(self))
        for i in range(len(self)):
            start, end = self._fragments.indptr[i], self._fragments.indptr[i + 1]
            if end > start:
                row_int = self._fragments.data[start:end]

                # Shannon Entropy: p_i = I_i / sum(I)
                p = row_int / np.sum(row_int)
                entropies[i] = -np.sum(p * np.log(p + 1e-12))
            else:
                entropies[i] = np.nan

        stats = pd.DataFrame({
            "peak_counts": peak_counts,
            "intensity_sums": intensity_sums,
            "intensity_entropy": entropies,
        }).describe()

        stats.attrs["label"] = "SpectraCollection Describe"
        stats.attrs["num_spectra"] = len(self)

        # Represent values in Jupyter nicely
        def _repr_html_():
            return stats.style.format({
                "peak_counts": "{:,.2f}",
                "intensity_sums": "{:,.0f}",
                "intensity_entropy": "{:.2f}"
            }).to_html()

        stats._repr_html_ = _repr_html_

        return stats


class MetadataProxy(pd.DataFrame):
    """
    Metadata proxy class.
    Used for filter directly on metadata and synchronize fragments.
    """
    _metadata = ["_collection"]

    def __init__(self, data, collection=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        object.__setattr__(self, "_collection", collection)

    @property
    def _constructor(self):
        return MetadataProxy

    def sort_values(self, by, inplace=False, **kwargs):
        result = self._collection.sort(by=by, inplace=inplace, **kwargs)
        return None if inplace else result.metadata


class FragmentsProxy:
    def __init__(self, csr_array):
        self._array = csr_array

    def sum(self, axis=1, **kwargs):
        result = self._array.sum(axis=axis, **kwargs)
        if axis == 1:
            return result.A1 if hasattr(result, "A1") else np.asarray(result).flatten()
        return result

    def max(self, axis: int = 1, **kwargs):
        return self._array.max(axis=axis, **kwargs)

    def min(self, axis: int = 1, **kwargs):
        return self._array.min(axis=axis, **kwargs)

    def mean(self, axis: int = 1, **kwargs):
        return self._array.mean(axis=axis, **kwargs)

    def count(self, axis: int = 1):
        if axis == 1:
            return np.diff(self._array.indptr)

        elif axis == 0:
            return np.bincount(self._array.indices, minlength=self._array.shape[1])

        else:
            raise ValueError("axis must be 0 or 1")

    def __getattr__(self, name):
        return getattr(self._array, name)

    def __getitem__(self, key):
        return self._array[key]

    def __repr__(self):
        return self._array.__repr__()
