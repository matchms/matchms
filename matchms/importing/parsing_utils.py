"""Helper functions for parsing metadata.
"""
from typing import Any
from typing import Union


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

    return {"charge": charge,
            "scan_number": scan_number,
            "title": title,
            "precursor_mz": precursor_mz,
            "scan_start_time": scan_time,
            "retention_time": retention_time}
