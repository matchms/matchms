from matchms import SpectrumProcessor
from matchms.filtering.default_pipelines import ALL_FILTER_SETS


def test_run_predefined_filter_sets():
    """Tests if all predefined filter sets can be run"""
    for filter_set in ALL_FILTER_SETS:
        SpectrumProcessor(filter_set)
