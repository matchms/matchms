import pytest
from matchms.filtering import correct_charge
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{}, 0],
        [{"ionmode": "positive"}, 1],
        [{"ionmode": "positive", "charge": -2}, 2],
        [{"ionmode": "negative", "charge": 2}, -2],
    ],
)
def test_correct_charge(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = correct_charge(spectrum_in)

    assert spectrum.get("charge") == expected


def test_correct_charge_empty_spectrum():
    spectrum_in = None
    spectrum = correct_charge(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
