import pytest
from testfixtures import LogCapture
from matchms.filtering import add_compound_name
from matchms.logging_functions import reset_matchms_logger
from .builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("metadata, expected, check_log", [
    [{"name": "Testospectrum"}, "Testospectrum", False],
    [{"title": "Testospectrum"}, "Testospectrum", False],
    [{"othername": "Testospectrum"}, None, True]
])
def test_add_compound_name(metadata, expected, check_log):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    with LogCapture() as log:
        spectrum = add_compound_name(spectrum_in)

    assert spectrum.get(
        "compound_name") == expected, "Expected no compound name."
    if check_log:
        log.check(('matchms', 'WARNING', 'No compound name found in metadata.'))
        reset_matchms_logger()


def test_empty_spectrum():
    spectrum_in = None
    spectrum = add_compound_name(spectrum_in)

    assert spectrum is None, "Expected differnt handling of None spectrum."
