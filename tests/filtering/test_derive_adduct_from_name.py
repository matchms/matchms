import pytest
from testfixtures import LogCapture
from matchms.filtering import derive_adduct_from_name
from matchms.filtering.metadata_processing.derive_adduct_from_name import _looks_like_adduct
from matchms.logging_functions import reset_matchms_logger, set_matchms_logger_level
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, remove_adduct_from_name, expected_adduct, expected_name, removed_adduct",
    [
        [{"compound_name": "peptideXYZ [M+H+K]"}, True, "[M+H+K]", "peptideXYZ", "[M+H+K]"],
        [{"compound_name": "GalCer(d18:2/16:1); [M+H]+"}, True, "[M+H]+", "GalCer(d18:2/16:1)", "[M+H]+"],
        [{"compound_name": "peptideXYZ [M+H+K]", "adduct": "M+H"}, True, "M+H", "peptideXYZ", "[M+H+K]"],
        [{"compound_name": "peptideXYZ [M+H+K]"}, False, "[M+H+K]", "peptideXYZ [M+H+K]", None],
        [{"Name": "peptideXYZ [M+H+K]", "adduct": "M+H"}, True, "M+H", "peptideXYZ", "[M+H+K]"],
        [{"compound_name": "peptideXYZ [M+H+K] C16H12"}, True, "[M+H+K]", "peptideXYZ C16H12", "[M+H+K]"],
        [{"name": ""}, True, None, "", None],
    ],
)
def test_derive_adduct_from_name_parametrized(metadata: dict, remove_adduct_from_name: bool, expected_adduct, expected_name, removed_adduct):
    set_matchms_logger_level("INFO")
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    with LogCapture() as log:
        spectrum = derive_adduct_from_name(spectrum_in, remove_adduct_from_name=remove_adduct_from_name)

    assert spectrum.get("adduct") == expected_adduct, "Expected different adduct."
    assert spectrum.get("compound_name") == expected_name, "Expected different cleaned name."

    expected_log = []
    if spectrum.get("compound_name") != spectrum_in.get("compound_name"):
        expected_log.append(("matchms", "INFO", f"Removed adduct ['{removed_adduct}'] from compound name."))
    if spectrum.get("adduct") != spectrum_in.get("adduct"):
        expected_log.append(("matchms", "INFO", f"Added adduct {expected_adduct} from the compound name to metadata."))

    log.check(*expected_log)
    reset_matchms_logger()


@pytest.mark.parametrize(
    "metadata, expected_adduct, expected_name",
    [
        # Test removing two adducts and selecting the interpretable one
        [{"compound_name": "peptideXYZ [M+H+K] C16H12 [M+H+K]2+"}, "[M+H+K]2+", "peptideXYZ C16H12"],
        # Test removing two adducts and not adding if not interpretable
        [{"compound_name": "peptideXYZ [M+H+K] C16H12 [M+2H+K]"}, None, "peptideXYZ C16H12"],
        # Test removing two adducts and check that clean adduct is run (to recover known adducts)
        [{"compound_name": "peptideXYZ [M+H+K] C16H12 [M+H]"}, "[M+H]+", "peptideXYZ C16H12"],
    ],
)
def test_derive_adduct_from_name_multiple_adducts_in_name(metadata, expected_adduct, expected_name):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = derive_adduct_from_name(spectrum_in, remove_adduct_from_name=True)

    assert spectrum.get("adduct") == expected_adduct, "Expected different adduct."
    assert spectrum.get("compound_name") == expected_name, "Expected different cleaned name."


def test_empty_spectrum():
    spectrum_in = None
    spectrum = derive_adduct_from_name(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."


def test_looks_like_adduct():
    """Test if adducts are correctly identified"""
    for adduct in ["M+", "M*+", "M+Cl", "[M+H]", "[2M+Na]+", "M+H+K", "[2M+ACN+H]+", "MS+Na", "MS+H", "M3Cl37+Na", "[M+H+H2O]"]:
        assert _looks_like_adduct(adduct), "Expected this to be identified as adduct"
    for adduct in ["N+", "B*+", "++", "--", "[--]", "H+M+K"]:
        assert not _looks_like_adduct(adduct), "Expected this not to be identified as adduct"
