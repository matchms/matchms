from __future__ import annotations
from functools import cached_property
from typing import Generator
import numpy as np
import pandas as pd
from matchms.MetadataTable import MetadataTable, harmonize_metadata_table_columns
from matchms.Spectrum import Spectrum
from .FragmentCollection import CSRFragmentCollection, FragmentCollection
from .hashing import compute_combined_hashes
from .typing import SpectraCollectionType


class SpectraCollection:
    """Central collection object for matchms spectra datasets.

    A ``SpectraCollection`` stores many spectra in a synchronized, table-like
    representation. It separates spectrum-level metadata from peak data while
    preserving a shared row order between both components.

    This class synchronizes:

    - metadata, tabular data kept internally as pandas ``DataFrame``
    - fragments, stored in a fragment backend, currently ``CSRFragmentCollection``

    Rows correspond to spectra. Metadata row ``i`` and fragment row ``i`` always
    describe the same spectrum. Operations such as slicing, filtering, sorting,
    dropping, and deduplication are applied to both metadata and fragments so that
    this alignment is preserved.

    Compared with a plain ``list[Spectrum]``, this representation is intended to
    support efficient collection-level operations, including metadata-based
    filtering, fragment-based filtering, m/z range slicing, sorting, hashing, and
    summary statistics.

    Individual rows can still be accessed as regular ``Spectrum`` objects. These
    objects are reconstructed from the stored metadata row and the corresponding
    fragment row.

    Notes
    -----
    The fragment backend may use an internal representation that differs from the
    original input spectra. In particular, the default CSR backend stores fragments
    as a binned sparse matrix. Reconstructed spectra therefore contain m/z values
    derived from the backend representation, for example bin centers, rather than
    necessarily the exact original input m/z values.

    The central invariant of this class is:

    ``len(metadata) == len(fragments) == n_spectra``

    and for every row index ``i``:

    ``metadata.iloc[i]`` corresponds to ``fragments.get_row(i)``.

    Direct modifications of internal metadata or fragment storage should be avoided.
    Use collection-level methods such as ``filter``, ``sort``, ``drop``, and
    ``add_metadata`` to preserve row alignment and invalidate cached values
    correctly.
    """
    def __init__(
        self,
        spectra: list[Spectrum] | Generator[Spectrum, None, None],
        bin_size=0.000001,
    ):
        spectra = list(spectra)

        if not spectra:
            raise ValueError("Spectra must contain at least one Spectrum.")

        self.bin_size = bin_size
        self._metadata = self._construct_metadata(spectra)
        self._fragments = self._construct_fragments(spectra)

        if len(self._metadata) != self._fragments.shape[0]:
            raise ValueError("Spectra Metadata/Fragments mismatch.")

    @classmethod
    def _from_metadata_and_fragments(
        cls,
        metadata: pd.DataFrame | pd.Series,
        fragments: FragmentCollection,
        bin_size: float,
    ) -> SpectraCollectionType:
        if isinstance(metadata, pd.Series):
            metadata = metadata.to_frame().T

        obj = cls.__new__(cls)
        obj.bin_size = bin_size
        obj._metadata = metadata.reset_index(drop=True)
        obj._fragments = fragments
        return obj

    def _construct_fragments(self, spectra: list):
        return CSRFragmentCollection(spectra, bin_size=self.bin_size)

    def _construct_metadata(self, spectra):
        # data = defaultdict(list)
        # [data[k].append(v) for spectrum in spectra for k, v in spectrum.metadata.items()]
        # TODO: add minimal Matadata harmonization

        # create and return pd.DataFrame(data)
        records = [spectrum.metadata for spectrum in spectra]
        metadata = pd.DataFrame.from_records(records)
        if len(metadata) == 0:  # allow empty metadata if spectra have no metadata
            metadata = pd.DataFrame(index=np.arange(len(spectra)))

        return metadata.reset_index(drop=True)

    @property
    def metadata(self) -> pd.DataFrame:
        return MetadataTable(self._metadata, self)

    @property
    def fragments(self) -> FragmentCollection:
        return self._fragments

    @property
    def n_spectra(self):
        return self._fragments.shape[0]
    
    @property
    def n_metadata_columns(self):
        return self._metadata.shape[1]

    @property
    def n_bins(self):
        return self._fragments.shape[1]

    def _normalize_row_selection(self, idx):
        """Normalize row selection to integer indices or a scalar int."""
        if isinstance(idx, (int, np.integer)):
            return int(idx)

        if isinstance(idx, slice):
            return np.arange(len(self))[idx]

        arr = np.asarray(idx)
        if arr.dtype == bool:
            if arr.shape[0] != len(self):
                raise ValueError(
                    f"Shape of row selector ({arr.shape[0]}) does not fit Items in SpectraCollection ({len(self)})."
                )
            return np.where(arr)[0]

        return arr.astype(np.int64)

    def _spectrum_from_row(self, idx: int) -> Spectrum:
        mz, intensities = self._fragments.get_row(int(idx))
        return Spectrum(
            mz=mz,
            intensities=intensities,
            metadata=self._metadata.iloc[int(idx)].to_dict(),
            metadata_harmonization=False,
        )

    def __getitem__(self, idx):
        # 2D slicing: rows + mz-range
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise IndexError("Expected at most two indexers: rows, mz-range")

            row_sel, mz_sel = idx

            # scalar row + mz slice -> one Spectrum
            if isinstance(row_sel, (int, np.integer)):
                row_idx = int(row_sel)
                new_fragments = self._fragments[[row_idx], mz_sel]
                mz, intensities = new_fragments.get_row(0)
                return Spectrum(
                    mz=mz,
                    intensities=intensities,
                    metadata=self._metadata.iloc[row_idx].to_dict(),
                    metadata_harmonization=False,
                )

            row_indices = self._normalize_row_selection(row_sel)
            new_metadata = self._metadata.iloc[row_indices].reset_index(drop=True)
            new_fragments = self._fragments[row_indices, mz_sel]

            return self.__class__._from_metadata_and_fragments(
                metadata=new_metadata,
                fragments=new_fragments,
                bin_size=self.bin_size,
            )

        # scalar row -> one Spectrum
        if isinstance(idx, (int, np.integer)):
            return self._spectrum_from_row(int(idx))

        # row-only selection -> SpectraCollection
        indices = self._normalize_row_selection(idx)
        target = self.copy()
        return target._reorder(indices)

    def __iter__(self):
        for i in range(self._fragments.shape[0]):
            yield self[i]

    def __len__(self):
        return self._fragments.shape[0]

    def __repr__(self) -> str:
        return (
            f"SpectraCollection(n_spectra={len(self)}, "
            f"n_metadata_columns={self._metadata.shape[1]}, "
            f"fragments={self._fragments!r})"
        )

    def __str__(self):
        return self.__repr__()

    @cached_property
    def spectra_hashes(self):
        return compute_combined_hashes(self.fragment_hashes, self.metadata_hashes)

    @cached_property
    def fragment_hashes(self):
        return self._fragments.fragment_hashes

    @cached_property
    def metadata_hashes(self):
        return pd.util.hash_pandas_object(self._metadata, index=False).tolist()

    def add_metadata(self, data, col_name: str = None, overwrite: bool = False):
        # TODO: must contain same sorting as present spectra/metadata. Add class bool flag, if data has been sorted?
        if isinstance(data, pd.DataFrame):
            new_metadata = data.copy()
        elif isinstance(data, pd.Series):
            col_name = col_name or data.name
            if col_name is None:
                raise ValueError("Series must have a name or 'col_name' must be provided.")
            new_metadata = pd.DataFrame({col_name: data})
        elif isinstance(data, list):
            if col_name is None:
                raise ValueError("'col_name' must be provided.")
            new_metadata = pd.DataFrame({col_name: data})
        elif isinstance(data, dict):
            for v in data.values():
                if not isinstance(v, list):
                    raise ValueError("When data is a dict, values must be of type list.")
            new_metadata = pd.DataFrame(data)
        else:
            raise TypeError("Data must be pd.DataFrame, pd.Series, list, or dict of lists.")

        if new_metadata.shape[0] != len(self):
            raise ValueError("New metadata does not match length of existing metadata entries.")

        overlap = self._metadata.columns.intersection(new_metadata.columns)
        if not overlap.empty:
            if not overwrite:
                raise ValueError(f"Columns already exist: {list(overlap)}. Set overwrite to True to replace values.")

            self._metadata = self._metadata.drop(columns=overlap)

        new_metadata = new_metadata.reset_index(drop=True)
        self._metadata = pd.concat(
            [self._metadata.reset_index(drop=True), new_metadata],
            axis=1,
        )
        self._clear_cache(["metadata_hashes"])

    def harmonize_metadata_columns(self, inplace: bool = False):
        """Harmonize metadata column names to matchms key style."""
        target = self if inplace else self.copy()

        target._metadata = harmonize_metadata_table_columns(target._metadata).reset_index(
            drop=True
        )
        target._clear_cache(["metadata_hashes", "spectra_hashes"])

        return None if inplace else target

    def _reorder(self, indices: np.ndarray):
        self._fragments = self._fragments.take(indices)
        self._metadata = self._metadata.iloc[indices].reset_index(drop=True)
        self._clear_cache()

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
            raise ValueError(
                f"Shape of filter mask ({mask.shape[0]}) does not fit Items in SpectraCollection ({len(self)})."
            )

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

        all_indices = np.arange(len(target))
        keep_mask = ~np.isin(all_indices, indices)

        target._fragments = target._fragments.filter(keep_mask)
        target._metadata = target._metadata.iloc[keep_mask].reset_index(drop=True)

        target._clear_cache()

        return None if inplace else target

    def drop_empty_spectra(self, inplace: bool = False):
        """
        Removes spectra without peaks.

        Parameters:
        -----------
        inplace : bool
            Will return a new SpectraCollection, if True and the same if False. Defaults to False.
        """
        peaks_per_row = self._fragments.count(axis=1)
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
        """
        return self._fragments.mz_to_bin(mz)

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
        return self._fragments.bin_to_mz(bin_idx)

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
        peak_counts = self._fragments.count(axis=1)
        intensity_sums = np.asarray(self._fragments.sum(axis=1)).flatten()

        entropies = np.zeros(len(self))
        for i in range(len(self)):
            _, row_int = self._fragments.get_row(i)
            if len(row_int) > 0:
                # Shannon Entropy: p_i = I_i / sum(I)
                p = row_int / np.sum(row_int)
                entropies[i] = -np.sum(p * np.log(p + 1e-12))
            else:
                entropies[i] = np.nan

        stats = pd.DataFrame(
            {
                "peak_counts": peak_counts,
                "intensity_sums": intensity_sums,
                "intensity_entropy": entropies,
            }
        ).describe()

        stats.attrs["label"] = "SpectraCollection Describe"
        stats.attrs["num_spectra"] = len(self)

        # Represent values in Jupyter nicely
        def _repr_html_():
            return stats.style.format(
                {"peak_counts": "{:,.2f}", "intensity_sums": "{:,.0f}", "intensity_entropy": "{:.2f}"}
            ).to_html()

        stats._repr_html_ = _repr_html_

        return stats
