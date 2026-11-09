import os
from collections.abc import Generator
from pathlib import Path
from typing import TextIO
from pyteomics.mgf import MGF
from matchms.importing.collection_builder import build_spectra_collection
from matchms.importing.parsing_utils import parse_spectrum_dict
from matchms.spectra_collection import SpectraCollection
from matchms.spectrum import Spectrum


def load_from_mgf(
    filename: str | Path | TextIO, metadata_harmonization: bool = True
) -> Generator[Spectrum, None, None]:
    """Load spectrum(s) from mgf file.

    This function will create :class:`~matchms.spectrum.Spectrum` objects for every spectrum 
    in the given .mgf file (or the file-like object).

    Examples:

    .. code-block:: python

        from matchms.importing import load_from_mgf

        file_mgf = "pesticides.mgf"
        spectra_from_path = list(load_from_mgf(file_mgf))

        # Or you can read the file in your application
        with open(file_mgf, "r") as spectra_file:
            spectra_from_file = list(load_from_mgf(spectra_file))

    Parameters
    ----------
    filename:
        Accepts both filename (with path) for .mgf file or a file-like
        object from a preloaded MGF file.
    metadata_harmonization : bool, optional
        Set to False if metadata harmonization to default keys is not desired.
        The default is True.
    """
    filename = _check_mgf_source(filename)

    def parse_file():
        with MGF(filename, convert_arrays=1, encoding="utf-8") as reader:
            for pyteomics_spectrum in reader:
                yield parse_spectrum_dict(
                    spectrum=pyteomics_spectrum,
                    metadata_harmonization=metadata_harmonization,
                )

    return parse_file()


def load_collection_from_mgf(
    filename: str | Path | TextIO,
    metadata_harmonization: bool = True,
    *,
    mz_precision: float = 1e-6,
    mz_rounding: str = "round",
) -> SpectraCollection:
    """Load an mgf file directly into a :class:`~matchms.SpectraCollection`.

    The result is the same as ``SpectraCollection(load_from_mgf(filename))``:
    the same metadata table and the same fragment array. The difference is
    how it gets there. Records from pyteomics are written straight into the
    collection's metadata table and packed peak arrays, so no intermediate
    :class:`~matchms.spectrum.Spectrum` objects are created and metadata keys
    are harmonized once per distinct key instead of once per spectrum. For
    large libraries this is many times faster and needs less memory.

    Parameters
    ----------
    filename:
        Path to the .mgf file or a file-like object with mgf content.
    metadata_harmonization:
        Set to False if metadata harmonization to default keys is not desired.
        The default is True.
    mz_precision:
        m/z grid precision of the collection's fragment backend. Default 1e-6.
    mz_rounding:
        m/z binning mode of the fragment backend, ``"round"`` or ``"floor"``.
    """
    filename = _check_mgf_source(filename)

    def records():
        with MGF(filename, convert_arrays=1, encoding="utf-8") as reader:
            for pyteomics_spectrum in reader:
                yield (
                    pyteomics_spectrum.get("params") or {},
                    pyteomics_spectrum["m/z array"],
                    pyteomics_spectrum["intensity array"],
                )

    return build_spectra_collection(
        records(),
        metadata_harmonization=metadata_harmonization,
        mz_precision=mz_precision,
        mz_rounding=mz_rounding,
    )


def _check_mgf_source(filename: str | Path | TextIO) -> str | TextIO:
    if isinstance(filename, str) and not os.path.isfile(filename):
        raise FileNotFoundError(f"The specified file: {filename} doesn't exist.")

    if isinstance(filename, Path):
        if not filename.is_file():
            raise FileNotFoundError(f"The specified file: {filename} doesn't exist.")
        return str(filename)  # pyteomics does not support pathlib.Path
    return filename
