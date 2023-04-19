import csv
import json
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from ..Spectrum import Spectrum


def _get_metadata_dict(spectrum: Spectrum, include_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Extract keys from spectrum metadata. Will silently continue if a key is not found.

    Args:
        spectrum (Spectrum): Spectrum with metadata to extract.
        include_fields (List[str] | str): "all" or set of metadata keys to extract.

    Returns:
        dict[str, Any]: Dictionary containing the specified keys.
    """
    if include_fields is None or include_fields[0] == "all":
        return spectrum.metadata
    if not isinstance(include_fields, list):
        print("'Include_fields' must be 'all' or list of keys.")
        return None

    return {key: spectrum.metadata[key] for key in spectrum.metadata.keys()
            & include_fields}


def export_metadata_as_json(spectrums: List[Spectrum], filename: str,
                            include_fields: Optional[List[str]] = None):
    """Export metadata to json file.

    Parameters
    ----------
    spectrums:
        Expected input is a list of  :py:class:`~matchms.Spectrum.Spectrum` objects.
    filename:
        Provide filename to save metadata of spectrum(s) as json file.
    identifier:
        Identifier used for naming each spectrum in the output file.
    """
    metadata_dicts = []
    for spec in spectrums:
        metadata_dict = _get_metadata_dict(spec, include_fields)
        if metadata_dict:
            metadata_dicts.append(metadata_dict)

    with open(filename, 'w', encoding="utf-8") as fout:
        json.dump(metadata_dicts, fout)


def export_metadata_as_csv(spectra: List[Spectrum], filename: str,
                           include_fields: Optional[List[str]] = None):
    """Export metadata to csv file.

    Parameters
    ----------
    spectra:
        Expected input is a list of  :py:class:`~matchms.Spectrum.Spectrum` objects.
    filename:
        Provide filename to save metadata of spectrum(s) as csv file.
    identifier:
        Identifier used for naming each spectrum in the output file.
    """
    metadata, columns = get_metadata_as_array(spectra)

    if include_fields is not None:
        metadata, columns = _subset_metadata(include_fields, metadata, columns)

    with open(filename, 'a', encoding="utf-8") as csvfile:  #TODO: assert if file exists
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        for data in metadata:
            writer.writerow(data)


def _subset_metadata(include_fields: List[str], metadata: np.array, columns: Set[str]) -> Tuple[np.array, Set[str]]:
    """Subset metadata to 'include_fields' and return intersection of columns.

    Parameters
    ----------
    include_fields:
        Columns to include.
    metadata:
        Data to subset.
    columns:
        Set of columns present in data

    Returns:
        Tuple[np.array, set[str]]: Subset data and columns.
    """
    return metadata[include_fields], columns.intersection(include_fields)


def get_metadata_as_array(spectra: List[Spectrum]) -> Tuple[np.array, List[str]]:
    """Extract union of all metadata as numpy array from all spectra.

    Parameters
    ----------
    spectra:
        Spectra from which to collect metadata.

    Returns:
        Tuple[np.array, List[str]]: Metadata and union of all columns detected in all spectra.
    """
    keys = spectra[0].metadata.keys()
    for s in spectra:
        keys |= s.metadata.keys()

    values = []

    for s in spectra:
        value = tuple((s.get(k) for k in keys))
        values.append(value)

    values_array = np.array(values, dtype=[(k, np.chararray) for k in keys])
    return values_array, keys
