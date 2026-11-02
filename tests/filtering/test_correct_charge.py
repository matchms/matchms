import numpy as np
import pytest
from matchms import SpectrumProcessor
from matchms.filtering import correct_charge
from matchms.filtering.default_pipelines import DEFAULT_FILTERS
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{}, 0],
        [{"ionmode": "positive"}, 1],
        [{"ionmode": "positive", "charge": -2}, 2],
        [{"ionmode": "negative", "charge": 2}, -2],
        [{"ionmode": "negative", "adduct": "[M-2H]2-"}, -2],
        [{"ionmode": "positive", "adduct": "[M+2H]2+", "charge": 0}, 2],
        [{"ionmode": "positive", "adduct": "M+2H"}, 2],
        [{"adduct": "[M+H]+"}, 1],
        [{"ionmode": "positive", "adduct": "[M+H]+", "charge": 3}, 3],
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


def test_correct_charge_multiply_charged_adduct_in_default_pipeline():
    # MassBank MSP records give Precursor_type and Ion_mode but no charge
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(np.array([100.0, 150.0, 200.0]))
        .with_intensities(np.array([0.1, 1.0, 0.5]))
        .with_metadata({"precursor_mz": 500.25, "ionmode": "negative", "adduct": "[M-2H]2-"})
        .build()
    )
    spectrum = SpectrumProcessor(DEFAULT_FILTERS).process_spectrum(spectrum_in)

    assert spectrum.get("charge") == -2
    assert spectrum.get("parent_mass") == pytest.approx(2 * 500.25 + 2 * 1.007276, abs=1e-3)
