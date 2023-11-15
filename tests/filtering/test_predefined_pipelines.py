from matchms import SpectrumProcessor
from matchms.filtering.default_pipelines import PREDEFINED_PIPELINES


def test_create_predefined_pipelines():
    """Tests if all predefined pipelines can be run"""
    for pipeline_name in PREDEFINED_PIPELINES:
        SpectrumProcessor(pipeline_name)

