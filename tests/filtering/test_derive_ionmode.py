import pytest
from matchms.filtering import derive_ionmode
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "adduct, charge, ionmode, expected_ionmode",
    [
        ["[M+H]", 1, None, "positive"],
        ["[M+H]", 1, "blabla", "positive"],
        [None, None, "blabla", "blabla"],
        ["M-H-", -1, None, "negative"],
        ["M+H", None, None, "positive"],
        ["blabla", 3, None, "positive"],
    ],
)
def test_derive_ionmode(adduct, charge, ionmode, expected_ionmode, as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "adduct": adduct,
                "charge": charge,
                "ionmode": ionmode,
            }
        )
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        derive_ionmode,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("ionmode") == expected_ionmode, "Expected different ionmode."


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_derive_ionmode_existing_valid_ionmode_is_kept(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "adduct": "[M-H]-",
                "charge": -1,
                "ionmode": "positive",
            }
        )
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        derive_ionmode,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("ionmode") == "positive"


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_derive_ionmode_conflicting_charge_and_adduct_keeps_original_ionmode(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "adduct": "[M-H]-",
                "charge": 1,
                "ionmode": None,
            }
        )
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        derive_ionmode,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("ionmode") is None


def test_derive_ionmode_empty_spectrum():
    spectrum = derive_ionmode(None)

    assert spectrum is None, "Expected different handling of None spectrum."
