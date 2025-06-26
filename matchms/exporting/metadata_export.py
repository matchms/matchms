import csv
import json
import logging
from typing import Any, Optional
import numpy as np
from ..Spectrum import Spectrum
from ..utils import filter_empty_spectra, rename_deprecated_params


logger = logging.getLogger("matchms")


def _get_metadata_dict(
    spectrum: Spectrum, include_fields: Optional[list[str]] = None
) -> dict[str, Any]:
    """Extract keys from spectrum metadata. Will silently continue if a key is not found.

    Args:
        spectrum (Spectrum): Spectrum with metadata to extract.
        include_fields (list[str] | str): "all" or set of metadata keys to extract.

    Returns:
        dict[str, Any]: Dictionary containing the specified keys.
    """
    if include_fields is None or include_fields[0] == "all":
        return spectrum.metadata
    if not isinstance(include_fields, list):
        logger.warning("'Include_fields' must be 'all' or list of keys.")
        return None

    return {
        key: spectrum.metadata[key] for key in spectrum.metadata.keys() & include_fields
    }


@rename_deprecated_params(param_mapping={"spectrums": "spectra"}, version="0.26.5")
def export_metadata_as_json(
    spectra: list[Spectrum], filename: str, include_fields: Optional[list[str]] = None
):
    """Export metadata to json file.

    Parameters
    ----------
    spectra:
        Expected input is a list of  :py:class:`~matchms.Spectrum.Spectrum` objects.
    filename:
        Provide filename to save metadata of spectrum(s) as json file.
    include_fields:
        Columns to include.
    """
    spectra = filter_empty_spectra(spectra)
    metadata_dicts = []
    for spec in spectra:
        metadata_dict = _get_metadata_dict(spec, include_fields)
        if metadata_dict:
            metadata_dicts.append(metadata_dict)

    with open(filename, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(metadata_dicts, indent=2, sort_keys=True))


def export_metadata_as_csv(
    spectra: list[Spectrum],
    filename: str,
    include_fields: Optional[list[str]] = None,
    delimiter: str = ",",
):
    """Export metadata to csv file.

    Parameters
    ----------
    spectra:
        Expected input is a list of  :py:class:`~matchms.Spectrum.Spectrum` objects.
    filename:
        Provide filename to save metadata of spectrum(s) as csv file.
    include_fields:
        Columns to include.
    delimiter:
        delimiter to use in the csv file, default is comma (",").
    """
    spectra = filter_empty_spectra(spectra)
    metadata, columns = get_metadata_as_array(spectra)

    if include_fields is not None:
        metadata, columns = _subset_metadata(include_fields, metadata, columns)

    with open(filename, "a+", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        writer.writerow(columns)
        for data in metadata:
            writer.writerow(data)


def _subset_metadata(
    include_fields: list[str], metadata: np.array, columns: list[str]
) -> tuple[np.array, list[str]]:
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
        tuple[np.array, set[str]]: Subset data and columns.
    """
    keys_subset = sorted(set(columns).intersection(include_fields))
    data_subset = metadata[keys_subset]
    return data_subset, keys_subset


def get_metadata_as_array(spectra: list[Spectrum]) -> tuple[np.array, list[str]]:
    """Extract union of all metadata as numpy array from all spectra.

    Parameters
    ----------
    spectra:
        Spectra from which to collect metadata.

    Returns:
        tuple[np.array, list[str]]: Metadata and union of all columns detected in all spectra.
    """
    spectra = filter_empty_spectra(spectra)
    keys = _get_metadata_keys(spectra)

    values = []

    for s in spectra:
        value = tuple((s.get(k) for k in keys))
        values.append(value)

    # Create a structured numpy array with keys as field names
    # and values as the corresponding data.
    values_array = np.array(values, dtype=[(k, object) for k in keys])
    return values_array, keys


def _get_metadata_keys(spectra: list[Spectrum]) -> list[str]:
    """  Get union of all metadata keys from all spectra.

    Args:
        spectra (list[Spectrum]): List of spectra to extract metadata keys from.

    Returns:
        _type_: _description_
    """
    keys = spectra[0].metadata.keys()
    for s in spectra:
        keys |= s.metadata.keys()
    return sorted(keys)
