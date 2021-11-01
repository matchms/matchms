import csv
import json
from typing import List
from typing import Union
from ..Spectrum import Spectrum


def get_metadata_dict(include_fields):
    if include_fields == "all":
        return spec.metadata
    if not isinstance(include_fields, list):
        print("'Include_fields' must be 'all' or list of keys.")
        return None

    return {key: spec.metadata[key] for key in spec.metadata.keys()
            & include_fields}

  
def export_metadata_as_json(spectrums: List[Spectrum], filename: str,
                            include_fields: Union[List, str] = "all"):
    """Export metadata to json file.
    
    Parameters
    ----------
    spectrums:
        Expected input is a list of  :py:class:`~matchms.Spectrum.Spectrum` objects.
    filename:
        Provide filename to save metadata of spectrum(s) as json file.
    """
    metadata_dicts = []
    for spec in spectrums:
        metadata_dict = get_metadata_dict(include_fields)
        if metadata_dict:
            metadata_dicts.append(metadata_dict)

    with open(filename, 'w') as fout:
        json.dump(metadata_dicts, fout)


def export_metadata_as_csv(spectrums: List[Spectrum], filename: str,
                           include_fields: Union[List, str] = "all"):
    """Export metadata to csv file.
    
    Parameters
    ----------
    spectrums:
        Expected input is a list of  :py:class:`~matchms.Spectrum.Spectrum` objects.
    filename:
        Provide filename to save metadata of spectrum(s) as csv file.
    """
    metadata_dicts = []
    columns = set()
    for i, spec in enumerate(spectrums):
        metadata_dict = get_metadata_dict(include_fields)
        if metadata_dict:
            metadata_dicts.append(metadata_dict)
            if i == 0:
                columns = metadata_dict.keys()
            else:
                columns = columns.intersection(metadata_dict.keys())

    try:
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")
