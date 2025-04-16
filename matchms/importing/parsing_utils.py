"""Helper functions for parsing metadata."""

import ast
from typing import Any, Dict, Union
import numpy as np
from matchms.Spectrum import Spectrum


def find_by_key(data: Union[list, dict], target: str) -> Any:
    """Helper function to return entries from nested list/dictionary.

    Parameters
    ----------
    data:
        Nested dictionary or list in which entry should be searched.
    target:
        Name of field to search for in data.
    """
    if hasattr(data, "items"):
        for key, value in data.items():
            if key == target:
                yield value
            if isinstance(value, dict):
                yield from find_by_key(value, target)
            elif isinstance(value, list):
                for val in value:
                    yield from find_by_key(val, target)

    elif isinstance(data, list):
        for subdata in data:
            yield from find_by_key(subdata, target)


def parse_mzml_mzxml_metadata(spectrum_dict: dict) -> dict:
    """Parse relevant mzml (or mzxml) metadata entries.

    Parameters
    ----------
    spectrum_dict:
        Spectrum dictionary containing metadata fields. Metadata parsing may easily
        break when field key names vary. The following metadata information is considered
        here:

        - precursor_mz, searched for in:
            -->"precursor"/"precursorMz"--> ... --> "selected ion m/z"/"precursorMz"
        - charge, searched for in:
            --> "charge state"/"polarity"
        - title, searched for in "spectrum title"
        - scan_number, searched for in "num"
        - scan_start_time, searched for in "scan start time"
        - retention_time, searched for in "retentionTime"

    """
    charge = None
    title = None
    scan_number = None
    precursor_mz = None
    scan_time = None
    retention_time = None

    first_search = list(find_by_key(spectrum_dict, "precursor"))
    if not first_search:
        first_search = list(find_by_key(spectrum_dict, "precursorMz"))
    if first_search:
        precursor_mz_search = list(find_by_key(first_search, "selected ion m/z"))
        if not precursor_mz_search:
            precursor_mz_search = list(find_by_key(first_search, "precursorMz"))
        if precursor_mz_search:
            precursor_mz = float(precursor_mz_search[0])

    precursor_charge = list(find_by_key(first_search, "charge state"))
    if precursor_charge:
        charge = int(precursor_charge[0])
    elif "polarity" in spectrum_dict:
        if spectrum_dict["polarity"] == "-":
            charge = -1
        elif spectrum_dict["polarity"] == "+":
            charge = 1

    if "spectrum title" in spectrum_dict:
        title = spectrum_dict["spectrum title"]
    if "num" in spectrum_dict:
        scan_number = spectrum_dict["num"]

    scan_time = list(find_by_key(spectrum_dict, "scan start time"))
    retention_time = list(find_by_key(spectrum_dict, "retentionTime"))

    return {
        "charge": charge,
        "scan_number": scan_number,
        "title": title,
        "precursor_mz": precursor_mz,
        "scan_start_time": scan_time,
        "retention_time": retention_time,
    }


def sort_by_mz(mz, intensities):
    """Sort mz values and intensities by mz."""
    if not np.all(mz[:-1] <= mz[1:]):
        idx_sorted = np.argsort(mz)
        mz = mz[idx_sorted]
        intensities = intensities[idx_sorted]
    return mz, intensities


def parse_spectrum_dict(spectrum: Dict, metadata_harmonization, spectrum_type="pyteomics") -> Spectrum:
    """Parse a spectrum dict (as read from a msp file for instance) to a matchms Spectrum."""
    metadata = spectrum.get("params", None)
    mz = spectrum["m/z array"]
    intensities = spectrum["intensity array"]

    if spectrum_type == "pyteomics":
        if "peak_comments" in metadata.keys():
            metadata["peak_comments"] = ast.literal_eval(str(metadata["peak_comments"]))
    else:
        peak_comments = spectrum["peak comments"]
        if peak_comments != {}:
            metadata["peak_comments"] = peak_comments

    mz, intensities = sort_by_mz(mz=mz, intensities=intensities)

    return Spectrum(mz=mz, intensities=intensities, metadata=metadata, metadata_harmonization=metadata_harmonization)
