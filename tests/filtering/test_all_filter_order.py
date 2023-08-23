"""These tests are meant to check that certain conditions are maintained,
when changing the ALL_FILTERS order in the future"""
import ast
import os
from typing import List, Callable
import pytest
from matchms.filtering.filter_order_and_default_pipelines import ALL_FILTERS, PREDEFINED_PIPELINES
from matchms.filtering.SpectrumProcessor import SpectrumProcessor
from matchms import filtering as msfilters

REPAIR_PARENT_MASS_SMILES_FILTERS = \
    [msfilters.repair_smiles_of_salts, msfilters.repair_precursor_is_parent_mass,
     msfilters.repair_parent_mass_is_mol_wt, msfilters.repair_adduct_based_on_smiles,
     msfilters.repair_parent_mass_match_smiles_wrapper, ]


@pytest.mark.parametrize("early_filters, later_filters", [
    [[msfilters.repair_not_matching_annotation], [msfilters.require_valid_annotation]],
    [REPAIR_PARENT_MASS_SMILES_FILTERS, [msfilters.require_parent_mass_match_smiles]],
])
def test_all_filter_order(early_filters: List[Callable], later_filters: List[Callable]):
    """Tests if early_filter is run before later_filter"""
    for early_filter in early_filters:
        for later_filter in later_filters:
            for filter_index, filter_function in enumerate(ALL_FILTERS):
                if early_filter == filter_function:
                    early_filter_index = filter_index
                if later_filter == filter_function:
                    later_filter_index = filter_index
            assert later_filter_index > early_filter_index, \
                f"The filter {early_filter.__name__} should be before {later_filter.__name__}"


def test_all_filters_is_complete():
    """Checks that the global varible ALL_FILTERS contains all the available filters

    This is important, since performing tests in the wrong order can make some filters useless.
    """
    def get_functions_from_file(file_path):
        """Gets all python functions in a file"""
        with open(file_path, 'r', encoding="utf-8") as file:
            tree = ast.parse(file.read(), filename=file_path)
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        return functions

    current_dir = os.path.dirname(os.path.abspath(__file__))
    filtering_directory = os.path.join(current_dir, "../../matchms/filtering")
    directories_with_filters = ["metadata_processing",
                                "peak_processing"]

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
        assert filter_function in all_filters, \
            f"The function {filter_function} in the script {script} is not given in ALL_FILTERS, " \
            f"this should be added to ensure a correct order of filter functions." \
            f"If this function is not a filter add a _ before the function name"


def test_all_filters_no_duplicates():
    all_filters = [filter.__name__ for filter in ALL_FILTERS]
    assert len(all_filters) == len(set(all_filters)), "One of the filters appears twice in ALL_FILTERS"


def test_create_predefined_pipelines():
    """Tests if all predefined pipelines can be run"""
    for pipeline_name in PREDEFINED_PIPELINES:
        SpectrumProcessor(pipeline_name)
