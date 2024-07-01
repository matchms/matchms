from matchms import SpectrumProcessor
from matchms.filtering.default_pipelines import ALL_FILTER_SETS


def test_run_predefined_filter_sets():
    """Tests if all predefined filter sets can be run"""
    for filter_set in ALL_FILTER_SETS:
        SpectrumProcessor(filter_set)


def test_all_filters_can_handle_none():
    """All filters should be able to take None as input instead of a spectrum object"""
    for filter_set in ALL_FILTER_SETS:
        spectrum_processor = SpectrumProcessor(filter_set)
        for selected_filter in spectrum_processor.filters:
            assert selected_filter(None) is None, f"The filter {selected_filter.__name__} cannot hande None as input"
