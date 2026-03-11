from typing import Generator
import numpy as np
import pandas as pd
from scipy.sparse import coo_array, csr_array
from matchms.Spectrum import Spectrum


class SpectraCollection:
    def __init__(self, spectra: list[Spectrum] | Generator[Spectrum, None, None], bin_size=0.000001):
        spectra = list(spectra)

        self._metadata = self._construct_metadata(spectra)
        self._fragments = self._construct_fragments(spectra, bin_size)
        self.bin_size = bin_size

        if len(self._metadata) != self._fragments.shape[0]:
            raise ValueError("Spectra Metadata/Fragments mismatch.")

    def _construct_fragments(self, spectra: list, bin_size: float = 0.000001):
        all_mz = np.concatenate([spec.mz for spec in spectra])
        all_int = np.concatenate([spec.intensities for spec in spectra])

        bin_idx = np.floor(all_mz / bin_size).astype(int)
        bin_no = bin_idx.max() + 1

        lengths = np.array([len(spec.mz) for spec in spectra])
        row_idx = np.repeat(np.arange(len(spectra)), lengths)

        return coo_array((all_int, (row_idx, bin_idx)), shape=(len(spectra), bin_no)).tocsr()  # CSR here!

    def _construct_metadata(self, spectra):
        # data = defaultdict(list)
        # [data[k].append(v) for spectrum in spectra for k, v in spectrum.metadata.items()]

        # return pd.DataFrame(data)
        records = [spectrum.metadata for spectrum in spectra]
        return pd.DataFrame.from_records(records)

    @property
    def metadata(self) -> pd.DataFrame:
        return MetadataProxy(self._metadata, self)

    @property
    def fragments(self) -> csr_array:
        return FragmentsProxy(self._fragments)

    @property
    def shape(self):
        return self._fragments.shape[0], self._metadata.shape[1]

    def __getitem__(self, idx):
        # csr = self.fragments.tocsr()
        csr = self._fragments

        start, end = csr.indptr[idx], csr.indptr[idx + 1]
        cols = csr.indices[start:end]
        intensities = csr.data[start:end]
        mz = cols * self.bin_size

        return Spectrum(mz=mz, intensities=intensities, metadata=self._metadata.iloc[idx].to_dict())

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

    def _reorder(self, indices: np.ndarray):
        self._fragments = self._fragments[indices, :]
        self._metadata = self._metadata.iloc[indices].reset_index(drop=True)

        return self


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

    def sort_values(self, by, **kwargs):
        kwargs.pop("inplace", None)
        sorted_df = super().sort_values(by=by, **kwargs)

        if self._collection is not None:
            self._collection._reorder(sorted_df.index.values)

        return MetadataProxy(self._collection._metadata, self._collection)


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
