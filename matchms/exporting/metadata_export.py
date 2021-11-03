import csv
import json
from typing import List
from typing import Union
from ..Spectrum import Spectrum


def get_metadata_dict(spectrum: Spectrum, include_fields: Union[List, str]):
    if include_fields == "all":
        return spectrum.metadata
    if not isinstance(include_fields, list):
        print("'Include_fields' must be 'all' or list of keys.")
        return None

    return {key: spectrum.metadata[key] for key in spectrum.metadata.keys()
            & include_fields}


def export_metadata_as_json(spectrums: List[Spectrum], filename: str,
                            include_fields: Union[List, str] = "all",
                            identifier: Union[None, str] = None):
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
        metadata_dict = get_metadata_dict(spec, include_fields)
        if metadata_dict:
            metadata_dicts.append(metadata_dict)

    with open(filename, 'w', encoding="utf-8") as fout:
        json.dump(metadata_dicts, fout)


def export_metadata_as_csv(spectrums: List[Spectrum], filename: str,
                           include_fields: Union[List, str] = "all",
                        identifier: Union[None, str] = None):
    """Export metadata to csv file.

    Parameters
    ----------
    spectrums:
        Expected input is a list of  :py:class:`~matchms.Spectrum.Spectrum` objects.
    filename:
        Provide filename to save metadata of spectrum(s) as csv file.
    identifier:
        Identifier used for naming each spectrum in the output file.
    """
    metadata_dicts = []
    columns = set()
    for i, spec in enumerate(spectrums):
        metadata_dict = get_metadata_dict(spec, include_fields)
        if metadata_dict:
            metadata_dicts.append(metadata_dict)
            if i == 0:
                columns = metadata_dict.keys()
            else:
                columns = columns.intersection(metadata_dict.keys())

    for metadata_dict in metadata_dicts:
        try:
            with open(filename, 'a', encoding="utf-8") as csvfile:  #TODO: assert if file exists
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                for data in metadata_dict[columns]:
                    writer.writerow(data)
        except IOError:
            print("I/O error")
