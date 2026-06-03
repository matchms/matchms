import pytest
from testfixtures import LogCapture
from matchms.filtering import add_compound_name
from matchms.logging_functions import reset_matchms_logger, set_matchms_logger_level
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "metadata, expected, check_log",
    [
        [{"name": "Testospectrum"}, "Testospectrum", False],
        [{"title": "Testospectrum"}, "Testospectrum", False],
        [{"othername": "Testospectrum"}, None, True],
    ],
)
def test_add_compound_name(metadata, expected, check_log, as_collection):
    set_matchms_logger_level("INFO")
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    with LogCapture() as log:
        spectrum = run_filter_as_spectrum_or_collection(
            add_compound_name,
            spectrum_in,
            as_collection,
        )

    assert spectrum.get("compound_name") == expected, "Expected different compound name."

    if check_log:
        log.check(("matchms", "INFO", "No compound name found in metadata."))

    reset_matchms_logger()


def test_add_compound_name_empty_spectrum():
    assert add_compound_name(None) is None
