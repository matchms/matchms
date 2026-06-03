import pytest
from matchms import SpectraCollection
from matchms.filtering import require_compound_name
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{"compound_name": "Acephate"}, SpectrumBuilder().with_metadata({"compound_name": "Acephate"}).build()],
        [{"formula": "H2O"}, None],
        [{}, None],
    ],
)
def test_require_compound_name(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_compound_name(spectrum_in)

    assert spectrum == expected, "Expected no changes."


def test_require_compound_name_collection():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"compound_name": "Acephate"}).build(),
            SpectrumBuilder().with_metadata({"formula": "H2O"}).build(),
            SpectrumBuilder().with_metadata({}).build(),
            SpectrumBuilder().with_metadata({"compound_name": "Glucose"}).build(),
        ]
    )

    filtered = require_compound_name(collection)

    assert filtered is not collection
    assert len(filtered) == 2
    assert filtered.metadata["compound_name"].tolist() == ["Acephate", "Glucose"]
