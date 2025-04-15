import pytest
from matchms.filtering import harmonize_undefined_inchikey
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, aliases, undefined, expected",
    [
        [{"inchikey": ""}, ["", "N/A", "NA", "n/a", "no data"], "", ""],
        [{"inchikey": "n/a"}, ["", "N/A", "NA", "n/a", "no data"], "", ""],
        [{"inchikey": "N/A"}, ["", "N/A", "NA", "n/a", "no data"], "", ""],
        [{"inchikey": "NA"}, ["", "N/A", "NA", "n/a", "no data"], "", ""],
        [{"inchikey": "no data"}, ["", "N/A", "NA", "n/a", "no data"], "", ""],
        [{"inchikey": "nan"}, ["nodata", "NaN", "Nan", "nan"], "", ""],
        [{"inchikey": "nan"}, ["nodata", "NaN", "Nan", "nan"], "n/a", "n/a"],
    ],
)
def test_harmonize_undefined_inchikey(metadata, aliases, undefined, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = harmonize_undefined_inchikey(spectrum_in, aliases=aliases, undefined=undefined)
    assert spectrum.get("inchikey") == expected


def test_empty_spectrum():
    spectrum_in = None
    spectrum = harmonize_undefined_inchikey(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
