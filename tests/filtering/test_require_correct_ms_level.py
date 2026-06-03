import pytest
from matchms import SpectraCollection
from matchms.filtering.metadata_processing.require_correct_ms_level import require_correct_ms_level
from ..builder_Spectrum import SpectrumBuilder


TEST_CASES = [
    ({"ms_level": "2"}, 2, False),
    ({"ms_level": "MS2"}, 2, False),
    ({}, 2, True),
    ({"ms_level": "MS3"}, 2, True),
    ({"ms_level": "MS2"}, 3, True),
    ({"ms_type": "MS2"}, 2, False),  # Check that key conversions are used
]


@pytest.mark.parametrize(
    "metadata, ms_level_to_keep, spectrum_removed",
    TEST_CASES,
)
def test_require_correct_ms_level_spectrum(metadata, ms_level_to_keep, spectrum_removed):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_correct_ms_level(spectrum_in, ms_level_to_keep)

    if spectrum_removed:
        assert spectrum is None, "Expected spectrum to be filtered out since it does not have the correct ms level"
    else:
        assert spectrum == spectrum_in


@pytest.mark.parametrize(
    "metadata, ms_level_to_keep, spectrum_removed",
    TEST_CASES,
)
def test_require_correct_ms_level_collection_single_row(metadata, ms_level_to_keep, spectrum_removed):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    collection = SpectraCollection([spectrum_in])

    filtered = require_correct_ms_level(collection, ms_level_to_keep)

    assert isinstance(filtered, SpectraCollection)
    assert len(filtered) == int(not spectrum_removed)


def test_require_correct_ms_level_collection_multiple_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"ms_level": "MS2"}).build(),
            SpectrumBuilder().with_metadata({"ms_level": "MS3"}).build(),
            SpectrumBuilder().with_metadata({}).build(),
            SpectrumBuilder().with_metadata({"ms_type": "MS2"}).build(),
        ]
    )

    filtered = require_correct_ms_level(collection, required_ms_level=2)

    assert filtered is not collection
    assert len(filtered) == 2
    assert filtered.metadata["ms_level"].tolist() == ["MS2", "MS2"]