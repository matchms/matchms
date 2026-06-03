import pytest
from matchms import SpectraCollection
from matchms.filtering.metadata_processing.require_matching_adduct_and_ionmode import (
    require_matching_adduct_and_ionmode,
)
from tests.builder_Spectrum import SpectrumBuilder


TEST_CASES = [
    ["positive", "[M+H]+", True],
    ["positive", "[M-H]-", False],
    ["negative", "[M-H]-", True],
    ["negative", "[M+H]+", False],
    ["negative", "bladiebla", False],
    [None, "[M+H]+", False],
]


@pytest.mark.parametrize(
    "ionmode, adduct, spectrum_kept",
    TEST_CASES,
)
def test_require_matching_adduct_and_ionmode_spectrum(ionmode, adduct, spectrum_kept):
    spectrum = SpectrumBuilder().with_metadata({"ionmode": ionmode, "adduct": adduct}).build()

    result = require_matching_adduct_and_ionmode(spectrum)

    if result is None:
        assert spectrum_kept is False
    else:
        assert spectrum_kept is True
        assert result == spectrum


@pytest.mark.parametrize(
    "ionmode, adduct, spectrum_kept",
    TEST_CASES,
)
def test_require_matching_adduct_and_ionmode_collection_single_row(ionmode, adduct, spectrum_kept):
    spectrum = SpectrumBuilder().with_metadata({"ionmode": ionmode, "adduct": adduct}).build()
    collection = SpectraCollection([spectrum])

    filtered = require_matching_adduct_and_ionmode(collection)

    assert isinstance(filtered, SpectraCollection)
    assert len(filtered) == int(spectrum_kept)


def test_require_matching_adduct_and_ionmode_collection_multiple_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"ionmode": "positive", "adduct": "[M+H]+"}).build(),
            SpectrumBuilder().with_metadata({"ionmode": "positive", "adduct": "[M-H]-"}).build(),
            SpectrumBuilder().with_metadata({"ionmode": "negative", "adduct": "[M-H]-"}).build(),
            SpectrumBuilder().with_metadata({"ionmode": None, "adduct": "[M+H]+"}).build(),
        ]
    )

    filtered = require_matching_adduct_and_ionmode(collection)

    assert filtered is not collection
    assert len(filtered) == 2
    assert filtered.metadata["ionmode"].tolist() == ["positive", "negative"]
    assert filtered.metadata["adduct"].tolist() == ["[M+H]+", "[M-H]-"]