"""These tests are meant to check that certain conditions are maintained,
when changing the ALL_FILTERS order in the future"""

import ast
import os
from typing import Callable, List
import pytest
from matchms import filtering as msfilters
from matchms.filtering.filter_order import ALL_FILTERS


REPAIR_PARENT_MASS_SMILES_FILTERS = [
    msfilters.repair_smiles_of_salts,
    msfilters.repair_parent_mass_is_molar_mass,
    msfilters.repair_adduct_and_parent_mass_based_on_smiles,
    msfilters.repair_adduct_based_on_parent_mass,
    msfilters.repair_parent_mass_match_smiles_wrapper,
]
DERIVE_ANNOTATION_FILTERS = [
    msfilters.derive_smiles_from_inchi,
    msfilters.derive_inchi_from_smiles,
    msfilters.derive_inchikey_from_inchi,
]


@pytest.mark.parametrize(
    "early_filters, later_filters",
    [
        [[msfilters.repair_not_matching_annotation], [msfilters.require_valid_annotation]],
        [REPAIR_PARENT_MASS_SMILES_FILTERS, [msfilters.require_parent_mass_match_smiles]],
        [
            DERIVE_ANNOTATION_FILTERS
            + [
                msfilters.repair_inchi_inchikey_smiles,
                msfilters.clean_adduct,
                msfilters.derive_annotation_from_compound_name,
            ],
            REPAIR_PARENT_MASS_SMILES_FILTERS,
        ],
        [
            [
                msfilters.add_precursor_mz,
            ],
            [
                msfilters.require_precursor_mz,
            ],
        ],
        # Since pubchem lookup checks if annotation is complete.
        # So deriving inchi and inchikey from smiles, should happen first.
        [
            [msfilters.repair_inchi_inchikey_smiles, msfilters.add_parent_mass],
            [
                msfilters.derive_annotation_from_compound_name,
            ],
        ],
        # The parent mass is based on the adduct, so adduct filters should be performed first
        [[msfilters.clean_adduct, msfilters.derive_adduct_from_name], [msfilters.add_parent_mass]],
        # The adduct filter removes all occurances while derive formula from name only removes when it is at the end
        # of compound name. Removing adducts therefore has to happen first.
        [[msfilters.derive_adduct_from_name], [msfilters.derive_formula_from_name]],
        [
            [
                msfilters.make_charge_int,
                msfilters.correct_charge,
            ],
            [msfilters.clean_adduct],
        ],
        [
            [
                msfilters.derive_adduct_from_name,
            ],
            [msfilters.clean_adduct],
        ],
        [
            [
                msfilters.derive_annotation_from_compound_name,
            ],
            DERIVE_ANNOTATION_FILTERS,
        ],
        [
            [
                msfilters.derive_formula_from_name,
            ],
            [msfilters.require_formula],
        ],
        [
            [
                msfilters.remove_profiled_spectra,
            ],
            [msfilters.remove_precursor_mz],
        ],
        [[msfilters.derive_formula_from_smiles], [msfilters.require_formula]],
        [
            [msfilters.require_valid_annotation] + REPAIR_PARENT_MASS_SMILES_FILTERS,
            [msfilters.derive_formula_from_smiles],
        ],
        [
            [
                msfilters.remove_profiled_spectra,
            ],
            [msfilters.remove_precursor_mz],
        ],
        [
            [msfilters.remove_noise_below_frequent_intensities],
            [
                msfilters.select_by_intensity,
                msfilters.select_by_mz,
                msfilters.select_by_relative_intensity,
                msfilters.remove_precursor_mz,
                msfilters.remove_peaks_outside_top_k,
                msfilters.reduce_to_number_of_peaks,
                msfilters.require_minimum_number_of_peaks,
                msfilters.require_minimum_number_of_high_peaks,
            ],
        ],
        [
            [
                msfilters.remove_profiled_spectra,
            ],
            [msfilters.remove_precursor_mz],
        ],
        [
            [
                msfilters.clean_adduct,
                msfilters.derive_adduct_from_name,
                msfilters.repair_adduct_based_on_parent_mass,
                msfilters.repair_adduct_and_parent_mass_based_on_smiles,
                msfilters.add_precursor_mz,
                msfilters.require_precursor_mz,
                msfilters.add_parent_mass,
                msfilters.repair_parent_mass_is_molar_mass,
            ],
            [msfilters.require_matching_adduct_precursor_mz_parent_mass],
        ],
        [
            [
                msfilters.repair_adduct_based_on_parent_mass,
                msfilters.repair_adduct_and_parent_mass_based_on_smiles,
                msfilters.clean_adduct,
                msfilters.require_correct_ionmode,
                msfilters.derive_ionmode,
                msfilters.derive_adduct_from_name,
            ],
            [msfilters.require_matching_adduct_and_ionmode],
        ],
        [
            [
                msfilters.remove_profiled_spectra,
            ],
            [msfilters.remove_precursor_mz],
        ],
        [
            [msfilters.repair_parent_mass_from_smiles],
            [
                msfilters.repair_adduct_based_on_parent_mass,
                msfilters.require_parent_mass_match_smiles,
                msfilters.require_matching_adduct_precursor_mz_parent_mass,
            ],
        ],
        [[msfilters.require_valid_annotation], [msfilters.repair_parent_mass_from_smiles]],
        [
            [
                msfilters.require_matching_adduct_precursor_mz_parent_mass,
                msfilters.derive_formula_from_smiles,
                msfilters.repair_adduct_based_on_parent_mass,
                msfilters.require_matching_adduct_and_ionmode,
            ],
            [msfilters.add_precursor_formula],
        ],
    ],
)
def test_all_filter_order(early_filters: List[Callable], later_filters: List[Callable]):
    """Tests if early_filter is run before later_filter"""
    for early_filter in early_filters:
        for later_filter in later_filters:
            early_filter_index = None
            later_filter_index = None

            for filter_index, filter_function in enumerate(ALL_FILTERS):
                if early_filter == filter_function:
                    early_filter_index = filter_index
                if later_filter == filter_function:
                    later_filter_index = filter_index

            assert early_filter_index is not None, f"{early_filter.__name__} not found in ALL_FILTERS"
            assert later_filter_index is not None, f"{later_filter.__name__} not found in ALL_FILTERS"
            assert later_filter_index > early_filter_index, (
                f"The filter {early_filter.__name__} should be before {later_filter.__name__}"
            )


def test_all_filters_is_complete():
    """Checks that the global varible ALL_FILTERS contains all the available filters

    This is important, since performing tests in the wrong order can make some filters useless.
    """

    def get_functions_from_file(file_path):
        """Gets all python functions in a file"""
        with open(file_path, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read(), filename=file_path)
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        return functions

    current_dir = os.path.dirname(os.path.abspath(__file__))
    filtering_directory = os.path.join(current_dir, "../../matchms/filtering")
    directories_with_filters = ["metadata_processing", "peak_processing"]

    all_filters = [filter.__name__ for filter in ALL_FILTERS]
    list_of_filter_function_names = []
    for directory_with_filters in directories_with_filters:
        directory_with_filters = os.path.join(filtering_directory, directory_with_filters)
        scripts = os.listdir(directory_with_filters)
        for script in scripts:
            # Remove __init__
            if script[0] == "_":
                break
            if script[-3:] == ".py":
                functions = get_functions_from_file(os.path.join(directory_with_filters, script))
                for function in functions:
                    if function[0] != "_":
                        list_of_filter_function_names.append((script, function))
    for script, filter_function in list_of_filter_function_names:
        assert filter_function in all_filters, (
            f"The function {filter_function} in the script {script} is not given in ALL_FILTERS, "
            f"this should be added to ensure a correct order of filter functions."
            f"If this function is not a filter add a _ before the function name"
        )


def test_all_filters_no_duplicates():
    all_filters = [filter.__name__ for filter in ALL_FILTERS]
    assert len(all_filters) == len(set(all_filters)), "One of the filters appears twice in ALL_FILTERS"
