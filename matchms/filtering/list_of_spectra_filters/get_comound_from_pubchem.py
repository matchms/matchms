import os
from typing import List
import numpy as np
import pandas as pd
import pubchempy as pcp
from tqdm import tqdm
from matchms import Spectrum
from matchms.metadata_utils import (is_valid_inchi, is_valid_inchikey,
                                    is_valid_smiles)


def pubchem_name_search(compound_name: str, name_search_depth=10,):
    """Search pubmed for compound name"""
    try:
        results_pubchem = pcp.get_compounds(compound_name,
                                            'name',
                                            listkey_count=name_search_depth)
        if len(results_pubchem) == 0 and "_" in compound_name:
            results_pubchem = pcp.get_compounds(compound_name.replace("_", " "),
                                                'name',
                                                listkey_count=name_search_depth)
        if len(results_pubchem) == 0:
            return []
    except (pcp.ServerError, ConnectionError, ConnectionAbortedError):
        print("Connection error, trying again")
        return pubchem_name_search(compound_name, name_search_depth=name_search_depth)
    return results_pubchem


def is_plausible_name(compound_name):
    return (isinstance(compound_name, str) and len(compound_name) > 4)


def select_unique_compound_names(compound_names):
    correct_compound_names = []
    for compound_name in compound_names:
        if is_plausible_name(compound_name):
            correct_compound_names.append(compound_name)
    return sorted(list(set(correct_compound_names)))


def check_fully_annotated(spectrum: Spectrum) -> bool:
    if not is_valid_smiles(spectrum.get("smiles")):
        return False
    if not is_valid_inchikey(spectrum.get("inchikey")):
        return False
    if not is_valid_inchi(spectrum.get("inchi")):
        return False
    return True


def select_unannotated_spectra(cleaned_spectra):
    unannotated_spectra = []
    annotated_spectra = []
    for spectrum in tqdm(cleaned_spectra, desc="Selecting not annotated_spectra"):
        if not check_fully_annotated(spectrum):
            unannotated_spectra.append(spectrum)
        else:
            annotated_spectra.append(spectrum)
    return unannotated_spectra, annotated_spectra


def write_compound_names_to_file(compound_names: list,
                                 csv_file):
    # Load in previously annotated compound names
    if os.path.exists(csv_file):
        already_annotated_compound_names = set(pd.read_csv(csv_file)["compound_name"])
    else:
        already_annotated_compound_names = set()
        with open(csv_file, "w", encoding="utf8") as f:
            f.write("compound_name,smiles,inchi,inchikey,monoisotopic_mass\n")
    unique_compound_names = select_unique_compound_names(compound_names)
    with open(csv_file, "a", encoding="utf8") as f:
        for compound_name in tqdm(unique_compound_names, "Retrieving compounds from pubchem"):
            if compound_name not in already_annotated_compound_names and is_plausible_name(compound_name):
                already_annotated_compound_names.add(compound_name)
                results = pubchem_name_search(compound_name)
                for result in results:
                    inchi_pubchem = '"' + result.inchi + '"'
                    inchikey_pubchem = result.inchikey
                    smiles_pubchem = result.isomeric_smiles
                    if smiles_pubchem is None:
                        smiles_pubchem = result.canonical_smiles
                    monoisotopic_mass = result.monoisotopic_mass
                    f.write(f"{compound_name},{smiles_pubchem},{inchi_pubchem},{inchikey_pubchem},{monoisotopic_mass}\n")
                if len(results) == 0:
                    f.write(f"{compound_name},,,,\n")
    return pd.read_csv(csv_file)


def find_pubchem_mass_match(compound_name_annotation: pd.DataFrame,
                            compound_name,
                            parent_mass,
                            mass_tolerance):
    """Selects matches based on compound class

    Parameters
    ----------
    compound_name_annotation: pd.DataFrame
        A dataframe with the compound names and information retrieved from pubchem
    compound_name: str
        Spectrum"s guessed parent mass.
    parent_mass: float
        The parent mass given for the spectrum
    mass_tolerance
        Acceptable mass difference between query compound and pubchem result.
    """
    if is_plausible_name(compound_name) and parent_mass is not None:
        matching_compound_name = compound_name_annotation[compound_name_annotation["compound_name"] == compound_name]
        mass_differences = np.abs(matching_compound_name["monoisotopic_mass"]-parent_mass)
        within_mass_tolerance = matching_compound_name[mass_differences < mass_tolerance]
        if within_mass_tolerance.shape[0] > 0:
            # Select the match with the most
            best_match = within_mass_tolerance.loc[within_mass_tolerance["monoisotopic_mass"].idxmin()]
            return best_match
    return None


def annotate_spectra(unannotated_spectra: List[Spectrum],
                     annotated_compound_names,
                     mass_tolerance=0.1):
    newly_annotated_spectra = []
    not_annotated_spectra = []
    for spectrum_in in tqdm(unannotated_spectra, desc="Annotating spectra based on pubchem compound name"):
        spectrum = spectrum_in.clone()
        result = find_pubchem_mass_match(annotated_compound_names,
                                         spectrum.get("compound_name"),
                                         spectrum.get("parent_mass"),
                                         mass_tolerance)
        if result is not None:
            spectrum.set("smiles", result["smiles"])
            spectrum.set("inchi", result["inchi"])
            spectrum.set("inchikey", result["inchikey"])
            newly_annotated_spectra.append(spectrum)
        else:
            not_annotated_spectra.append(spectrum)
    return newly_annotated_spectra, not_annotated_spectra


def annotate_with_pubchem_wrapper(spectra, compound_class_csv_file):
    """Annotates spectra which are not fully annotated based on the compound name
    """
    unannotated_spectra, annotated_spectra = select_unannotated_spectra(spectra)
    compound_names = [spectrum.get("compound_name") for spectrum in unannotated_spectra]
    annotated_compound_names = write_compound_names_to_file(compound_names,
                                                            compound_class_csv_file)
    newly_annotated_spectra, not_annotated_spectra = annotate_spectra(unannotated_spectra, annotated_compound_names)
    return annotated_spectra, newly_annotated_spectra, not_annotated_spectra
