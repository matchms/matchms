import numpy as np
import pytest
from matchms import SpectraCollection
from matchms.filtering import add_parent_mass
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{"pepmass": (444.0, 10), "charge": -1}, "Missing precursor m/z to derive parent mass."],
        [{"charge": -1}, "Missing precursor m/z to derive parent mass."],
        [{"precursor_mz": 444.0, "charge": 0}, "Not sufficient spectrum metadata to derive parent mass."],
    ],
)
def test_add_parent_mass_exceptions(metadata, expected, caplog, as_collection):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        add_parent_mass,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("parent_mass") is None, "Expected no parent mass."
    assert expected in caplog.text


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_add_parent_mass_precursormz(caplog, as_collection):
    """Test if parent mass is correctly derived if precursor_mz is present."""
    metadata = {"precursor_mz": 444.0, "charge": -1}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        add_parent_mass,
        spectrum_in,
        as_collection,
    )

    assert np.abs(spectrum.get("parent_mass") - 445.0) < 0.01, "Expected parent mass of about 445.0."
    assert isinstance(spectrum.get("parent_mass"), float), "Expected parent mass to be float."
    assert "Not sufficient spectrum metadata to derive parent mass." not in caplog.text


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize("overwrite, expected", [(True, 442.992724), (False, 443.0)])
def test_add_parent_mass_overwrite(overwrite, expected, as_collection):
    """Test if parent mass is replaced by newly calculated value."""
    metadata = {"precursor_mz": 444.0, "parent_mass": 443.0, "adduct": "[M+H]+", "charge": +1}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        add_parent_mass,
        spectrum_in,
        as_collection,
        overwrite_existing_entry=overwrite,
    )

    assert np.allclose(spectrum.get("parent_mass"), expected, atol=1e-4), (
        "Expected parent mass to be replaced by new value."
    )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "parent_mass_field, parent_mass, expected",
    [
        ("parent_mass", 400.0, 400.0),
        ("parent_mass", "400.", 400.0),
        ("exact_mass", 200, 200.0),
        ("parentmass", 200, 200.0),
        ("parent_mass", "n/a", 442.992724),
    ],
)
def test_add_parent_mass_already_present(parent_mass_field, parent_mass, expected, as_collection):
    """Test if parent mass is read from accepted existing fields."""
    metadata = {"precursor_mz": 444.0, parent_mass_field: parent_mass, "charge": +1}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        add_parent_mass,
        spectrum_in,
        as_collection,
    )

    assert np.allclose(spectrum.get("parent_mass"), expected, atol=1e-4), f"Expected parent mass of about {expected}."
    assert isinstance(spectrum.get("parent_mass"), (float, int)), "Expected parent mass to be numeric."


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_add_parent_mass_not_sufficient_data(caplog, as_collection):
    """Test when there is not enough information to derive parent_mass."""
    metadata = {"precursor_mz": 444.0}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        add_parent_mass,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("parent_mass") is None, "Expected no parent mass."
    assert "Not sufficient spectrum metadata to derive parent mass." in caplog.text


def test_add_parent_mass_empty_spectrum():
    assert add_parent_mass(None) is None


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "metadata, expected_parent_mass",
    [
        ({"smiles": "C"}, 16.031300),
        ({"exact_mass": 100, "smiles": "CH4"}, 100),  # Smiles should only be used if other options do not work.
        ({"precursor_mz": 10, "charge": 1, "smiles": "CH4"}, 8.9927235),
        ({"precursor_mz": 10, "adduct": "[M+H]+", "smiles": "CH4"}, 8.9927235),
    ],
)
def test_add_parent_mass_from_smiles(metadata, expected_parent_mass, as_collection):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        add_parent_mass,
        spectrum_in,
        as_collection,
    )

    assert np.allclose(spectrum.get("parent_mass"), expected_parent_mass, atol=1e-4), (
        f"Expected parent mass of about {expected_parent_mass}."
    )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize("estimate_from_charge, expected", [(True, 442.992724), (False, None)])
def test_add_parent_mass_not_from_charge(estimate_from_charge, expected, as_collection):
    metadata = {"precursor_mz": 444.0, "charge": +1}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        add_parent_mass,
        spectrum_in,
        as_collection,
        estimate_from_charge=estimate_from_charge,
    )

    if expected is not None:
        assert np.allclose(spectrum.get("parent_mass"), expected, atol=1e-4), (
            "Expected parent mass to be replaced by new value."
        )
    else:
        assert spectrum.get("parent_mass") is None


def test_add_parent_mass_collection_updates_multiple_rows():
    spectra = [
        SpectrumBuilder().with_metadata({"precursor_mz": 444.0, "charge": +1}).build(),
        SpectrumBuilder().with_metadata({"smiles": "C"}).build(),
        SpectrumBuilder().with_metadata({"precursor_mz": 444.0}).build(),
    ]
    collection = SpectraCollection(spectra)

    processed = add_parent_mass(collection)

    assert isinstance(processed, SpectraCollection)
    assert len(processed) == 3
    assert np.allclose(processed.metadata.loc[0, "parent_mass"], 442.992724, atol=1e-4)
    assert np.allclose(processed.metadata.loc[1, "parent_mass"], 16.031300, atol=1e-4)
    assert "parent_mass" not in processed.metadata.columns or np.isnan(processed.metadata.loc[2, "parent_mass"])


def test_add_parent_mass_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder()
            .with_metadata({"precursor_mz": 444.0, "charge": +1})
            .build()
        ]
    )

    processed = add_parent_mass(collection, clone=False)

    assert processed is collection
    assert np.allclose(collection.metadata.loc[0, "parent_mass"], 442.992724, atol=1e-4)
