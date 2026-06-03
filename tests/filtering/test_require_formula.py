import pytest
from matchms import SpectraCollection
from matchms.filtering import require_formula
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{"formula": "C6H12O6"}, SpectrumBuilder().with_metadata({"formula": "C6H12O6"}).build()],
        [{"formula": "Na2CO3"}, SpectrumBuilder().with_metadata({"formula": "Na2CO3"}).build()],
        [{"formula": "NaCl"}, SpectrumBuilder().with_metadata({"formula": "NaCl"}).build()],
        [{"formula": "H2O"}, SpectrumBuilder().with_metadata({"formula": "H2O"}).build()],
        [{"formula": "20C30H"}, None],
        [{}, None],
    ],
)
def test_require_formula_spectrum(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_formula(spectrum_in)

    assert spectrum == expected, "Expected no changes."


@pytest.mark.parametrize(
    "metadata, expected_kept",
    [
        [{"formula": "C6H12O6"}, True],
        [{"formula": "Na2CO3"}, True],
        [{"formula": "NaCl"}, True],
        [{"formula": "H2O"}, True],
        [{"formula": "20C30H"}, False],
        [{}, False],
    ],
)
def test_require_formula_collection_single_row(metadata, expected_kept):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    collection = SpectraCollection([spectrum_in])

    filtered = require_formula(collection)

    assert isinstance(filtered, SpectraCollection)
    assert len(filtered) == int(expected_kept)


def test_require_formula_collection_multiple_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"formula": "C6H12O6"}).build(),
            SpectrumBuilder().with_metadata({"formula": "20C30H"}).build(),
            SpectrumBuilder().with_metadata({}).build(),
            SpectrumBuilder().with_metadata({"formula": "NaCl"}).build(),
        ]
    )

    filtered = require_formula(collection)

    assert filtered is not collection
    assert len(filtered) == 2
    assert filtered.metadata["formula"].tolist() == ["C6H12O6", "NaCl"]