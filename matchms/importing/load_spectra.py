import os
from collections.abc import Generator, Iterator
from itertools import chain
from matchms.importing import (
    load_from_json,
    load_from_mgf,
    load_from_msp,
    load_from_mzml,
    load_from_mzxml,
    load_from_pickle,
)
from matchms.SpectraCollection import SpectraCollection
from matchms.Spectrum import Spectrum


def load_spectra(
    file: str,
    metadata_harmonization: bool = True,
    ftype: str | None = "auto",
) -> list[Spectrum] | Generator[Spectrum, None, None]:
    """Load spectra from a file as matchms Spectrum objects.

    The following file extensions can be loaded with this function:
    ``mzML``, ``json``, ``mgf``, ``msp``, ``mzxml`` and ``pickle``.

    A pickled file is expected to directly contain a list of matchms Spectrum
    objects.

    Parameters
    ----------
    file
        Path to file containing spectra.
    metadata_harmonization
        If True, harmonize metadata during import.
    ftype
        File type to use for import. By default, ``"auto"`` guesses the file
        type from the file extension. Alternatively, pass an explicit file type,
        for example ``"mzml"``, ``"json"``, ``"mgf"``, ``"msp"``, ``"mzxml"``,
        or ``"pickle"``.

    Returns
    -------
    list[Spectrum] or Generator[Spectrum, None, None]
        Imported spectra.
    """
    assert os.path.exists(file), f"The specified file: {file} does not exists"

    if ftype is None or ftype == "auto":
        ftype = os.path.splitext(file)[1].lower()[1:]
    else:
        ftype = ftype.lower()

    if ftype == "mzml":
        return load_from_mzml(file, metadata_harmonization=metadata_harmonization)
    if ftype == "json":
        return load_from_json(file, metadata_harmonization=metadata_harmonization)
    if ftype == "mgf":
        return load_from_mgf(file, metadata_harmonization=metadata_harmonization)
    if ftype == "msp":
        return load_from_msp(file, metadata_harmonization=metadata_harmonization)
    if ftype == "mzxml":
        return load_from_mzxml(file, metadata_harmonization=metadata_harmonization)
    if ftype == "pickle":
        return load_from_pickle(file, metadata_harmonization)

    raise TypeError(f"File extension of file: {file} is not recognized")


def load_ms2_dataset(
    file: str,
    metadata_harmonization: bool = True,
    ftype: str = "auto",
) -> SpectraCollection:
    """Load spectra from a file as a SpectraCollection.

    Parameters
    ----------
    file
        Path to file containing spectra.
    metadata_harmonization
        If True, harmonize metadata during import.
    ftype
        File type to use for import. By default, ``"auto"`` guesses the file
        type from the file extension. Alternatively, pass an explicit file type,
        for example ``"mzml"``, ``"json"``, ``"mgf"``, ``"msp"``, ``"mzxml"``,
        or ``"pickle"``.

    Returns
    -------
    SpectraCollection
        Imported spectra as a collection.
    """
    return SpectraCollection(
        load_spectra(
            file,
            metadata_harmonization=metadata_harmonization,
            ftype=ftype,
        )
    )


def load_list_of_spectrum_files(
    spectrum_files: list[str] | str,
) -> list[Spectrum] | Iterator[Spectrum]:
    """Combines all spectra in multiple files into a list of spectra"""
    # Just load spectra if it is a single file
    if isinstance(spectrum_files, str):
        return load_spectra(spectrum_files)
    # If multiple files combine results into one generator
    spectrum_generators = [load_spectra(spectrum_file) for spectrum_file in spectrum_files]

    return chain.from_iterable(spectrum_generators)
