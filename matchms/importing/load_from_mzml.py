import logging
from collections.abc import Generator
from pathlib import Path
import numpy as np
from pyteomics.mzml import read
from matchms.importing.parsing_utils import sort_by_mz
from matchms.spectrum import Spectrum


logger = logging.getLogger("matchms")


def load_from_mzml(
    filename: str | Path, ms_level: int = 2, metadata_harmonization: bool = True
) -> Generator[Spectrum, None, None]:
    """Load spectrum(s) from mzml file.

    This function will create :class:`~matchms.spectrum.Spectrum` objects for every spectrum of the
    desired ms_level found in a given MzML file. For more extensive parsing options consider
    using pyteomics or pymzml packages.

    Example:

    .. code-block:: python

        from matchms.importing import load_from_mzml

        file_mzml = "testdata.mzml"
        spectra = list(load_from_mzml(file_mzml))

    Parameters
    ----------
    filename:
        Filename for mzml file to import.
    ms_level:
        Specify which ms level to import. Default is 2.
    metadata_harmonization : bool, optional
        Set to False if metadata harmonization to default keys is not desired.
        The default is True.
    """
    if isinstance(filename, Path):
        filename = str(filename)  # pyteomics does not support pathlib.Path

    with read(filename, dtype=dict) as reader:
        for pyteomics_spectrum in reader:
            if "ms level" in pyteomics_spectrum and pyteomics_spectrum["ms level"] == ms_level:
                mz = np.asarray(pyteomics_spectrum.pop("m/z array"), dtype="float")
                intensities = np.asarray(pyteomics_spectrum.pop("intensity array"), dtype="float")

                mz, intensities = sort_by_mz(mz=mz, intensities=intensities)
                flattend_metadata = parse_metadata(pyteomics_spectrum)
                flattend_metadata = derive_charge_from_polarity(flattend_metadata)
                yield Spectrum(
                    mz=mz,
                    intensities=intensities,
                    metadata=flattend_metadata,
                    metadata_harmonization=metadata_harmonization,
                )


def parse_key_value(key, value, first_level=True):
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "count" and not first_level:
                continue
            if key is not None:
                combined_key = key + "-" + k
            else:
                combined_key = k
            yield from parse_key_value(combined_key, v, False)
    elif isinstance(value, list):
        if len(value) == 0:
            return
        if len(value) > 1:
            # Legal in mzML, e.g. a scanWindowList with count=2, or a cvParam
            # that occurs twice in one spectrum. matchms metadata is flat, so
            # only the first entry is kept.
            logger.warning(
                "Found %d values for '%s' in mzML spectrum, only the first is used.",
                len(value),
                key,
            )
        first = value[0]
        if isinstance(first, dict):
            for k, v in first.items():
                yield from parse_key_value(k, v, False)
        else:
            # list of plain values (repeated cvParam)
            yield key, first
    else:
        yield key, value


def derive_charge_from_polarity(flattend_metadata):
    """This is here for historic reasons, it would fit better in the filter correct_charge,
    but since the loader did this automatically before, we kept it here, to not break existing pipelines."""
    if "polarity" in flattend_metadata:
        if flattend_metadata["polarity"] == "-":
            flattend_metadata["charge"] = -1
        if flattend_metadata["polarity"] == "+":
            flattend_metadata["charge"] = 1
    return flattend_metadata


def parse_metadata(metadata_dict: dict):
    flattend_dict = {}
    for key, value in parse_key_value(None, metadata_dict):
        if key in flattend_dict:
            raise ValueError(f"The key:{key} is duplicated")
        flattend_dict[key] = value
    return flattend_dict
