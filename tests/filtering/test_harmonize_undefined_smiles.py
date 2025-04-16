import pytest
from matchms.filtering import harmonize_undefined_smiles
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, aliases, undefined, expected",
    [
        [{"smiles": ""}, ["", "N/A", "NA", "n/a", "no data"], "", ""],
        [{"smiles": "n/a"}, ["", "N/A", "NA", "n/a", "no data"], "", ""],
        [{"smiles": "N/A"}, ["", "N/A", "NA", "n/a", "no data"], "", ""],
        [{"smiles": "NA"}, ["", "N/A", "NA", "n/a", "no data"], "", ""],
        [{"smiles": "no data"}, ["", "N/A", "NA", "n/a", "no data"], "", ""],
        [{"smiles": "nan"}, ["nodata", "NaN", "Nan", "nan"], "", ""],
        [{"smiles": "nan"}, ["nodata", "NaN", "Nan", "nan"], "n/a", "n/a"],
    ],
)
def test_harmonize_undefined_smiles(metadata, aliases, undefined, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = harmonize_undefined_smiles(spectrum_in, aliases=aliases, undefined=undefined)
    assert spectrum.get("smiles") == expected


def test_empty_spectrum():
    spectrum_in = None
    spectrum = harmonize_undefined_smiles(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
