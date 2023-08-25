import csv
import logging
import os
from functools import lru_cache
from typing import List, Optional
import numpy as np
import pandas as pd
import pubchempy
from matchms import Spectrum
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import \
    _check_fully_annotated


logger = logging.getLogger("matchms")


def repair_smiles_from_compound_name(spectrum_in: Spectrum,
                                     annotated_compound_names_file: Optional[str] = None,
                                     mass_tolerance: float = 0.1):
    """Adds annotations (smiles, inchi, inchikey) based on compound name

    Based on a table of compound names and smiles matches (stored in a csv file) this function
    adds the new annotations to the input spectrums if the smiles seem consistent with the available
    spectrum metadata (e.g., parent mass).
    This function can be used with csv files that are returned by the pubchem_lookup.py
    from matchmextras.

    Parameters
    ----------
    spectrum_in:
        The input spectrum.
    annotated_compound_names_file: str
        A csv file containing the compound names and matching smiles, inchi, inchikey
        and monoisotopic_mass. This can be created using the the pubchem_lookup.py from matchmextras.
    mass_tolerance:
        Acceptable mass difference between query compound and pubchem result.
    """
    if spectrum_in is None:
        return None
    spectrum = spectrum_in.clone()
    if _check_fully_annotated(spectrum):
        return spectrum
    compound_name = spectrum.get("compound_name")
    parent_mass = spectrum.get('parent_mass')

    if _is_plausible_name(compound_name) and parent_mass is not None:

        compound_name_annotations = _get_compound_name_annotation(compound_name, annotated_compound_names_file)
        if len(compound_name_annotations) > 0:
            compound_name_annotation_df = pd.DataFrame(compound_name_annotations)
            mass_differences = np.abs(compound_name_annotation_df["monoisotopic_mass"] - parent_mass)
            within_mass_tolerance = compound_name_annotation_df[mass_differences < mass_tolerance]
            if within_mass_tolerance.shape[0] > 0:
                # Select the match with the most
                best_match = within_mass_tolerance.loc[within_mass_tolerance["monoisotopic_mass"].idxmin()]
                spectrum.set("smiles", best_match["smiles"])
                spectrum.set("inchi", best_match["inchi"])
                spectrum.set("inchikey", best_match["inchikey"])
                logger.info("Added smiles %s based on the compound name %s", best_match["smiles"], compound_name)
                return spectrum
    return spectrum


@lru_cache(maxsize=None)
def _get_compound_name_annotation(compound_name, csv_file=None) -> List[dict]:
    """Loads compound name annotation from file or gets it from pubchem any new annotation is added to the file

    functools.cache, makes sure that previously loaded or calculated compound names do not have to be reloaded.
    This reduces the number of time the csv file has to be loaded from memory.
    """
    if csv_file is None:
        return _pubchem_name_search(compound_name)

    # Search in previously annotated compound names
    annotated_compound_names = _load_compound_name_annotations(csv_file, compound_name)
    if not annotated_compound_names:
        annotated_compound_names = _pubchem_name_search(compound_name)
        if not annotated_compound_names:
            _write_compound_name_annotations(csv_file, [{"compound_name": compound_name, "smiles": None, "inchi": None,
                                                         "inchikey": None,  "monoisotopic_mass": None}])
        _write_compound_name_annotations(csv_file, annotated_compound_names)
    return annotated_compound_names


def _pubchem_name_search(compound_name: str, name_search_depth=10,) -> List[dict]:
    """Search pubmed for compound name"""
    try:
        results_pubchem = pubchempy.get_compounds(compound_name,
                                                  'name',
                                                  listkey_count=name_search_depth)
        if len(results_pubchem) == 0 and "_" in compound_name:
            results_pubchem = pubchempy.get_compounds(compound_name.replace("_", " "),
                                                      'name',
                                                      listkey_count=name_search_depth)
    except (pubchempy.ServerError, ConnectionError, ConnectionAbortedError, pubchempy.PubChemHTTPError):
        print("Connection error, trying again")
        return _pubchem_name_search(compound_name, name_search_depth=name_search_depth)

    extracted_results = []
    # extract the needed information:
    for result in results_pubchem:
        smiles_pubchem = result.isomeric_smiles
        if smiles_pubchem is None:
            smiles_pubchem = result.canonical_smiles
        extracted_results.append({"compound_name": compound_name,
                                  "smiles": smiles_pubchem,
                                  "inchi": result.inchi,
                                  "inchikey": result.inchikey,
                                  "monoisotopic_mass": float(result.monoisotopic_mass),})
    return extracted_results


def _load_compound_name_annotations(annotated_compound_names_csv, compound_name: str):
    """Loads in the annotated compound names and checks format"""
    if not os.path.exists(annotated_compound_names_csv):
        return []
    annotated_compound_names = pd.read_csv(annotated_compound_names_csv)
    assert list(annotated_compound_names.columns) == ["compound_name", "smiles", "inchi",
                                                      "inchikey", "monoisotopic_mass"], \
        "The annotated_compound_names_csv file does not have the columns " \
        "compound_name, smiles, inchi, inchikey, monoisotopic_mass"

    matches = annotated_compound_names[annotated_compound_names["compound_name"] == compound_name]
    matches.replace(np.nan, None, inplace=True)
    return matches.to_dict("records")


def _write_compound_name_annotations(annotated_compound_names_csv, compound_name_annotations: List[dict]):
    if not os.path.exists(annotated_compound_names_csv):
        with open(annotated_compound_names_csv, "w", encoding="utf8") as f:
            f.write("compound_name,smiles,inchi,inchikey,monoisotopic_mass\n")
    with open(annotated_compound_names_csv, "a", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for compound_name_annotation in compound_name_annotations:
            writer.writerow([compound_name_annotation["compound_name"],
                             compound_name_annotation["smiles"],
                             compound_name_annotation["inchi"],
                             compound_name_annotation["inchikey"],
                             compound_name_annotation["monoisotopic_mass"],])


def _is_plausible_name(compound_name):
    """Simple check if it can be a compound name."""
    return isinstance(compound_name, str) and len(compound_name) > 4
