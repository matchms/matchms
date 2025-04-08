import pytest
from matchms.filtering.metadata_processing.require_matching_adduct_and_ionmode import require_matching_adduct_and_ionmode
from tests.builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "ionmode, adduct, spectrum_kept",
    [
        ["positive", "[M+H]+", True],
        ["positive", "[M-H]-", False],
        ["negative", "[M-H]-", True],
        ["negative", "[M+H]+", False],
        ["negative", "bladiebla", False],
        [None, "[M+H]+", False],
    ],
)
def test_require_matching_adduct_and_ionmode(ionmode, adduct, spectrum_kept):
    spectrum = SpectrumBuilder().with_metadata({"ionmode": ionmode, "adduct": adduct}).build()
    result = require_matching_adduct_and_ionmode(spectrum)
    if result is None:
        assert spectrum_kept is False
    else:
        assert spectrum_kept is True
