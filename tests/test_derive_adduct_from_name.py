import pytest
from testfixtures import LogCapture
from matchms.filtering import derive_adduct_from_name
from matchms.logging_functions import reset_matchms_logger
from matchms.logging_functions import set_matchms_logger_level
from .builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("metadata, remove_adduct_from_name, expected_adduct, expected_name, removed_adduct", [
    [{"compound_name": "peptideXYZ [M+H+K]"}, True, "[M+H+K]", "peptideXYZ", "[M+H+K]"],
    [{"compound_name": "peptideXYZ [M+H+K]", "adduct": "M+H"}, True, "M+H", "peptideXYZ", "[M+H+K]"],
    [{"compound_name": "peptideXYZ [M+H+K]"}, False, "[M+H+K]", "peptideXYZ [M+H+K]", None],
    [{"name": ""}, True, None, None, None]
])
def test_derive_adduct_from_name_parametrized(metadata, remove_adduct_from_name, expected_adduct, expected_name, removed_adduct):
    set_matchms_logger_level("INFO")
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    with LogCapture() as log:
        spectrum = derive_adduct_from_name(spectrum_in, remove_adduct_from_name=remove_adduct_from_name)

    assert spectrum.get("adduct") == expected_adduct, "Expected different adduct."
    assert spectrum.get("compound_name") == expected_name, "Expected different cleaned name."

    expected_log = list()
    if spectrum.get("compound_name") != spectrum_in.get("compound_name"):
        expected_log.append(('matchms', 'INFO', f'Removed adduct {removed_adduct} from compound name.'))
    if spectrum.get("adduct") != spectrum_in.get("adduct"):
        expected_log.append(('matchms', 'INFO', f'Added adduct {expected_adduct} to metadata.'))

    log.check(*expected_log)
    reset_matchms_logger()


def test_empty_spectrum():
    spectrum_in = None
    spectrum = derive_adduct_from_name(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
