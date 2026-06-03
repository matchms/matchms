import pytest
from matchms import SpectraCollection
from matchms.filtering import require_correct_ionmode
from ..builder_Spectrum import SpectrumBuilder


TEST_CASES = [
    ("positive", "positive", False),
    ("negative", "negative", False),
    ("positive", "both", False),
    ("negative", "both", False),
    ("positive", "negative", True),
    ("negative", "positive", True),
    ("n/a", "both", True),
]


@pytest.mark.parametrize(
    "ionmode, ionmode_to_keep, spectrum_removed",
    TEST_CASES,
)
def test_require_correct_ionmode_spectrum(ionmode, ionmode_to_keep, spectrum_removed):
    spectrum_in = SpectrumBuilder().with_metadata({"ionmode": ionmode}).build()

    spectrum = require_correct_ionmode(spectrum_in, ionmode_to_keep)

    if spectrum_removed:
        assert spectrum is None, "Expected spectrum to be filtered out since it does not have the correct ionmode"
    else:
        assert spectrum == spectrum_in


@pytest.mark.parametrize(
    "ionmode, ionmode_to_keep, spectrum_removed",
    TEST_CASES,
)
def test_require_correct_ionmode_collection_single_row(ionmode, ionmode_to_keep, spectrum_removed):
    spectrum_in = SpectrumBuilder().with_metadata({"ionmode": ionmode}).build()
    collection = SpectraCollection([spectrum_in])

    filtered = require_correct_ionmode(collection, ionmode_to_keep)

    assert isinstance(filtered, SpectraCollection)
    assert len(filtered) == int(not spectrum_removed)


def test_require_correct_ionmode_collection_multiple_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"ionmode": "positive"}).build(),
            SpectrumBuilder().with_metadata({"ionmode": "negative"}).build(),
            SpectrumBuilder().with_metadata({"ionmode": "n/a"}).build(),
        ]
    )

    filtered = require_correct_ionmode(collection, "positive")

    assert filtered is not collection
    assert len(filtered) == 1
    assert filtered.metadata.loc[0, "ionmode"] == "positive"


def test_require_correct_ionmode_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"ionmode": "positive"}).build(),
            SpectrumBuilder().with_metadata({"ionmode": "negative"}).build(),
        ]
    )

    filtered = require_correct_ionmode(collection, "positive", clone=False)

    assert filtered is collection
    assert len(collection) == 1
    assert collection.metadata.loc[0, "ionmode"] == "positive"