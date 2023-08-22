"""These tests are meant to check that certain conditions are maintained,
when changing the ALL_FILTERS order in the future"""

import pytest
from matchms.SpectrumProcessor import ALL_FILTERS


@pytest.mark.parametrize("early_filter, later_filter", [
    ["require_valid_annotation", "repair_smiles_of_salts"],
])
def test_all_filter_order(early_filter, later_filter):
    """Tests if early_filter is run before later_filter"""
    for filter_index, filter_function in enumerate(ALL_FILTERS):
        filter_name = filter_function.__name__
        if early_filter == filter_name:
            early_filter_index = filter_index
        if later_filter == filter_name:
            later_filter_index = filter_index
    assert later_filter_index > early_filter_index, f"The filter {early_filter} should be before {later_filter}"
