import csv
import json
import logging
import os
import time
from functools import cache
from urllib.error import URLError
import numpy as np
import pandas as pd
import pubchempy
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    as_string_or_none,
)
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    is_valid_inchi,
    is_valid_inchikey,
    is_valid_smiles,
)


logger = logging.getLogger("matchms")


def _derive_annotation_from_compound_name(
    metadata,
    annotated_compound_names_file: str | None = None,
    mass_tolerance: float = 0.1,
) -> dict:
    """Add molecular annotations based on compound name by searching PubChem.

    This filter adds ``smiles``, ``inchi``, and/or ``inchikey`` metadata based on
    a PubChem compound-name lookup. SMILES lookup is not supported directly by
    pubchempy anymore, see https://github.com/matchms/matchms/issues/823.
    SMILES can alternatively be derived from InChI by running
    ``derive_smiles_from_inchi``.

    The filter is only run if there is not yet a valid SMILES or InChI entry in
    the metadata. The annotation is only added if the PubChem result has a
    monoisotopic mass close enough to the spectrum's ``parent_mass``.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
    annotated_compound_names_file
        Optional CSV file used as a persistent cache. Any compound name searched
        on PubChem will be added to this file. If a compound name is already
        present in the file, the cached annotation is used instead of querying
        PubChem again.

        The CSV file should contain the columns ``compound_name``, ``smiles``,
        ``inchi``, ``inchikey``, and ``monoisotopic_mass``.
    mass_tolerance
        Acceptable mass difference between query compound and PubChem result.
        Default is ``0.1``.
    clone
        Optionally clone the input before applying the filter. If ``False``, the
        input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Input object with added annotation metadata, or ``None`` if the input was
        ``None``.
    """
    if is_valid_inchi(as_string_or_none(metadata.get("inchi"))):
        return {}

    if is_valid_smiles(as_string_or_none(metadata.get("smiles"))):
        return {}

    compound_name = as_string_or_none(metadata.get("compound_name"))
    parent_mass = as_float_or_none(metadata.get("parent_mass"))

    if not _is_plausible_name(compound_name) or parent_mass is None:
        logger.info(
            "Could not find a matching annotation on PubChem for the compound name: %s",
            compound_name,
        )
        return {}

    compound_name_annotations = _get_pubchem_compound_name_annotation(
        compound_name,
        annotated_compound_names_file,
    )

    if len(compound_name_annotations) == 0:
        logger.info(
            "Could not find a matching annotation on PubChem for the compound name: %s",
            compound_name,
        )
        return {}

    annotation_df = pd.DataFrame(compound_name_annotations)
    annotation_df["monoisotopic_mass"] = pd.to_numeric(
        annotation_df["monoisotopic_mass"],
        errors="coerce",
    )

    annotation_df = annotation_df.dropna(subset=["monoisotopic_mass"])
    if annotation_df.empty:
        logger.info(
            "Could not find a matching annotation on PubChem for the compound name: %s",
            compound_name,
        )
        return {}

    mass_differences = np.abs(annotation_df["monoisotopic_mass"] - parent_mass)
    within_mass_tolerance = annotation_df[mass_differences < mass_tolerance]

    if within_mass_tolerance.empty:
        logger.info(
            "Could not find a matching annotation on PubChem for the compound name: %s",
            compound_name,
        )
        return {}

    best_match = annotation_df.loc[mass_differences.idxmin()]
    updates = {}

    smiles = as_string_or_none(best_match.get("smiles"))
    if is_valid_smiles(smiles):
        updates["smiles"] = smiles
        logger.info(
            "Added smiles %s based on the compound name %s",
            smiles,
            compound_name,
        )

    inchi = as_string_or_none(best_match.get("inchi"))
    if is_valid_inchi(inchi):
        updates["inchi"] = inchi
        logger.info(
            "Added inchi %s based on the compound name %s",
            inchi,
            compound_name,
        )

    inchikey = as_string_or_none(best_match.get("inchikey"))
    if is_valid_inchikey(inchikey):
        updates["inchikey"] = inchikey
        logger.info(
            "Added inchikey %s based on the compound name %s",
            inchikey,
            compound_name,
        )

    return updates


@cache
def _get_pubchem_compound_name_annotation(compound_name, csv_file=None) -> list[dict]:
    """Load compound-name annotation from file or retrieve it from PubChem.

    ``functools.cache`` ensures that previously loaded or calculated compound
    names do not have to be reloaded during the same Python session.
    """
    if csv_file is None:
        return _pubchem_name_search(compound_name)

    annotated_compound_names = _load_compound_name_annotations(
        csv_file,
        compound_name,
    )

    if annotated_compound_names:
        return annotated_compound_names

    annotated_compound_names = _pubchem_name_search(compound_name)

    if not annotated_compound_names:
        _write_compound_name_annotations(
            csv_file,
            [
                {
                    "compound_name": compound_name,
                    "smiles": None,
                    "inchi": None,
                    "inchikey": None,
                    "monoisotopic_mass": None,
                }
            ],
        )
        return []

    _write_compound_name_annotations(csv_file, annotated_compound_names)
    return annotated_compound_names


def _pubchem_name_search(
    compound_name: str,
    name_search_depth=10,
    max_retries=15,
) -> list[dict]:
    """Search PubChem for a compound name."""
    retries = 0
    max_delay = 3600

    while retries < max_retries:
        try:
            results_pubchem = pubchempy.get_compounds(
                compound_name,
                "name",
                listkey_count=name_search_depth,
            )

            if len(results_pubchem) == 0 and "_" in compound_name:
                results_pubchem = pubchempy.get_compounds(
                    compound_name.replace("_", " "),
                    "name",
                    listkey_count=name_search_depth,
                )

            extracted_results = []
            for result in results_pubchem:
                smiles_pubchem = result.smiles
                if smiles_pubchem is None:
                    smiles_pubchem = result.canonical_smiles

                extracted_results.append(
                    {
                        "compound_name": compound_name,
                        "smiles": smiles_pubchem,
                        "inchi": result.inchi,
                        "inchikey": result.inchikey,
                        "monoisotopic_mass": float(result.monoisotopic_mass),
                    }
                )

            return extracted_results

        except (
            pubchempy.ServerError,
            ConnectionError,
            ConnectionAbortedError,
            pubchempy.PubChemHTTPError,
            URLError,
            TimeoutError,
        ):
            retries += 1
            if retries >= max_retries:
                break

            delay = min(max_delay, 2**retries)
            logger.warning(
                "Connection error while querying PubChem for %s. "
                "Retrying after %s seconds.",
                compound_name,
                delay,
            )
            time.sleep(delay)

        except json.decoder.JSONDecodeError:
            logger.warning(
                "Compound name: %s resulted in broken json from PubChem",
                compound_name,
            )
            return []

    logger.error(
        "Compound name: %s could not be loaded due to a connection error after %s tries",
        compound_name,
        str(max_retries),
    )
    return []


def _load_compound_name_annotations(
    annotated_compound_names_csv,
    compound_name: str,
):
    """Load annotated compound names from CSV and check the expected format."""
    if not os.path.exists(annotated_compound_names_csv):
        return []

    annotated_compound_names = pd.read_csv(annotated_compound_names_csv)

    assert list(annotated_compound_names.columns) == [
        "compound_name",
        "smiles",
        "inchi",
        "inchikey",
        "monoisotopic_mass",
    ], (
        "The annotated_compound_names_csv file does not have the columns "
        "compound_name, smiles, inchi, inchikey, monoisotopic_mass"
    )

    matches = annotated_compound_names[
        annotated_compound_names["compound_name"] == compound_name
    ]

    return matches.to_dict("records")


def _write_compound_name_annotations(
    annotated_compound_names_csv,
    compound_name_annotations: list[dict],
):
    if not os.path.exists(annotated_compound_names_csv):
        with open(annotated_compound_names_csv, "w", encoding="utf8") as f:
            f.write("compound_name,smiles,inchi,inchikey,monoisotopic_mass\n")

    with open(annotated_compound_names_csv, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for compound_name_annotation in compound_name_annotations:
            writer.writerow(
                [
                    compound_name_annotation["compound_name"],
                    compound_name_annotation["smiles"],
                    compound_name_annotation["inchi"],
                    compound_name_annotation["inchikey"],
                    compound_name_annotation["monoisotopic_mass"],
                ]
            )


def _is_plausible_name(compound_name):
    """Return True if compound_name is a plausible compound name."""
    return isinstance(compound_name, str) and len(compound_name) > 4


derive_annotation_from_compound_name = metadata_update_filter(
    _derive_annotation_from_compound_name
)