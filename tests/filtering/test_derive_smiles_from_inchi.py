import pytest
from testfixtures import LogCapture
from matchms.filtering import derive_smiles_from_inchi
from matchms.logging_functions import reset_matchms_logger, set_matchms_logger_level
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_derive_smiles_from_inchi(as_collection):
    """Test if conversion to smiles works when only inchi is given."""
    set_matchms_logger_level("INFO")
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "inchi": '"InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2"',
                "smiles": "",
            }
        )
        .build()
    )

    with LogCapture() as log:
        spectrum = run_filter_as_spectrum_or_collection(
            derive_smiles_from_inchi,
            spectrum_in,
            as_collection,
        )

    assert spectrum.get("smiles") == "C1CCCCC1", "Expected different smiles"
    log.check(
        (
            "matchms",
            "INFO",
            "Added smiles C1CCCCC1 to metadata (was converted from InChI)",
        )
    )
    reset_matchms_logger()


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_derive_smiles_from_defect_inchi(as_collection):
    """Test if no smiles is derived from invalid inchi."""
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "inchi": '"InChI=1S/C6H12/c1-2-XA4-6-5-3-1/h1-6H2"',
                "smiles": "",
            }
        )
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        derive_smiles_from_inchi,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("smiles", None) == "", "Expected no smiles"


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_derive_smiles_from_inchi_does_not_overwrite_valid_smiles(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "inchi": '"InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2"',
                "smiles": "CCCO",
            }
        )
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        derive_smiles_from_inchi,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("smiles") == "CCCO"


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_derive_smiles_from_inchi_without_inchi_does_nothing(as_collection):
    spectrum_in = SpectrumBuilder().with_metadata({"smiles": ""}).build()

    spectrum = run_filter_as_spectrum_or_collection(
        derive_smiles_from_inchi,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("smiles", None) == ""


def test_derive_smiles_from_inchi_empty_spectrum():
    spectrum = derive_smiles_from_inchi(None)

    assert spectrum is None, "Expected different handling of None spectrum."
