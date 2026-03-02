import pytest
from matchms.filtering.metadata_processing.require_correct_ms_level import require_correct_ms_level
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, ms_level_to_keep, spectrum_removed",
    [
        ({"ms_level": "2"}, 2, False),
        ({"ms_level": "MS2"}, 2, False),
        ({}, 2, True),
        ({"ms_level": "MS3"}, 2, True),
        ({"ms_level": "MS2"}, 3, True),
        ({"ms_type": "MS2"}, 2, False),  # Check that key conversions are used
    ],
)
def test_require_correct_ms_level(metadata, ms_level_to_keep, spectrum_removed):
    builder = SpectrumBuilder()
    spectrum_in = builder.with_metadata(metadata).build()
    spectrum = require_correct_ms_level(spectrum_in, ms_level_to_keep)

    if spectrum_removed is True:
        assert spectrum is None, "Expected spectrum to be filtered out since it does not have the correct ionmode"
    else:
        assert spectrum == spectrum_in
