import pytest
from matchms.filtering import harmonize_undefined_inchi
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, aliases, undefined, expected",
    [
        [{"inchi": ""}, ["", "N/A", "NA", "n/a"], "", ""],
        [{"inchi": "n/a"}, ["", "N/A", "NA", "n/a"], "", ""],
        [{"inchi": "N/A"}, ["", "N/A", "NA", "n/a"], "", ""],
        [{"inchi": "NA"}, ["", "N/A", "NA", "n/a"], "", ""],
        [{"inchi": "nan"}, ["nodata", "NaN", "Nan", "nan"], "", ""],
        [{"inchi": "nan"}, ["nodata", "NaN", "Nan", "nan"], "n/a", "n/a"],
    ],
)
def test_harmonize_undefined_inchi(metadata, aliases, undefined, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = harmonize_undefined_inchi(spectrum_in, aliases=aliases, undefined=undefined)
    assert spectrum.get("inchi") == expected


def test_empty_spectrum():
    spectrum_in = None
    spectrum = harmonize_undefined_inchi(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
