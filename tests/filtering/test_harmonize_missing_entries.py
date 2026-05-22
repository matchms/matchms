import pytest
from matchms.filtering import harmonize_missing_entries
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_harmonize_missing_entries_all_keys(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "inchi": "n/a",
                "inchikey": "no data",
                "smiles": "C",
            }
        )
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        harmonize_missing_entries,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("inchi") is None
    assert spectrum.get("inchikey") is None
    assert spectrum.get("smiles") == "C"


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_harmonize_missing_entries_selected_keys(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "inchi": "n/a",
                "inchikey": "no data",
                "smiles": "no data",
            }
        )
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        harmonize_missing_entries,
        spectrum_in,
        as_collection,
        keys=["inchi", "inchikey"],
    )

    assert spectrum.get("inchi") is None
    assert spectrum.get("inchikey") is None
    assert spectrum.get("smiles") == "no data"


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_harmonize_missing_entries_custom_undefined(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata({"inchi": "n/a"})
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        harmonize_missing_entries,
        spectrum_in,
        as_collection,
        keys=["inchi"],
        undefined="",
    )

    assert spectrum.get("inchi") == ""


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_harmonize_missing_entries_creates_missing_selected_key(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata({"smiles": "C"})
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        harmonize_missing_entries,
        spectrum_in,
        as_collection,
        keys=["inchi"],
    )

    assert spectrum.get("inchi") is None
    assert spectrum.get("smiles") == "C"


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_harmonize_missing_entries_accepts_single_key_string(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata({"inchi": "n/a"})
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        harmonize_missing_entries,
        spectrum_in,
        as_collection,
        keys="inchi",
    )

    assert spectrum.get("inchi") is None
