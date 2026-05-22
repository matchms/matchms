import pytest
from testfixtures import LogCapture
from matchms.filtering import derive_inchikey_from_inchi
from matchms.logging_functions import reset_matchms_logger, set_matchms_logger_level
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_derive_inchikey_from_inchi(as_collection):
    """Test if conversion from inchi to inchikey works."""
    set_matchms_logger_level("INFO")
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "inchi": '"InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2"',
                "inchikey": "n/a",
            }
        )
        .build()
    )

    with LogCapture() as log:
        spectrum = run_filter_as_spectrum_or_collection(
            derive_inchikey_from_inchi,
            spectrum_in,
            as_collection,
        )

    assert spectrum.get("inchikey")[:14] == "XDTMQSROBMDMFD", "Expected different inchikey"
    log.check(
        (
            "matchms",
            "INFO",
            "Added InChIKey XDTMQSROBMDMFD-UHFFFAOYSA-N to metadata (was converted from inchi)",
        )
    )
    reset_matchms_logger()


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_derive_inchikey_from_inchi_does_not_overwrite_valid_inchikey(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "inchi": '"InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2"',
                "inchikey": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
            }
        )
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        derive_inchikey_from_inchi,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("inchikey") == "VNWKTOKETHGBQD-UHFFFAOYSA-N"


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_derive_inchikey_from_invalid_inchi_does_nothing(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "inchi": '"InChI=1S/C6H12/c1-2-XA4-6-5-3-1/h1-6H2"',
                "inchikey": "n/a",
            }
        )
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        derive_inchikey_from_inchi,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("inchikey") == "n/a"


def test_derive_inchikey_from_inchi_empty_spectrum():
    spectrum = derive_inchikey_from_inchi(None)

    assert spectrum is None, "Expected different handling of None spectrum."
