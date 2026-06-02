import logging
import pytest
from matchms import SpectraCollection
from matchms.filtering import add_precursor_formula
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "metadata, expected_formula",
    [
        [{"formula": "C12H15ClN2O3", "adduct": "[M+H]+"}, "C12H16ClN2O3"],
        [{"formula": "C12H15ClN2O3", "adduct": "[M+H-H2O]+"}, "C12H14ClN2O2"],  # multiple adducts
        [{"formula": "C12H15ClN2O3"}, None],  # no adduct
        [{"adduct": "[M+H-H2O]+"}, None],  # no formula
        [{"formula": "C2H8NO", "adduct": "[2M+H-H2O]+"}, "C4H15N2O"],
        [{"formula": "C2H6", "adduct": "[M-H2O]+"}, None],  # impossible adduct formula combo
        [{}, None],
    ],
)
def test_add_precursor_formula(metadata, expected_formula, as_collection):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        add_precursor_formula,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("precursor_formula") == expected_formula, "Expected different formula."


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "metadata, expected",
    [
        # Zero-cancellation: +Na -Na should yield the original formula with no Na present.
        ({"formula": "H2O", "adduct": "[M+Na-Na]+"}, "H2O"),
        # Base: C6H12O3, +2H => C6H14O3, +Na => C6H14NaO3, -3H2O => C6H8Na.
        ({"formula": "C6H12O3", "adduct": "[M+2H+Na-3H2O]+"}, "C6H8Na"),
        # Multiple parent masses: [3M+H]+
        # Base: C2H6O => 3M: C6H18O3, +H => C6H19O3.
        ({"formula": "C2H6O", "adduct": "[3M+H]+"}, "C6H19O3"),
    ],
)
def test_various_adduct_arithmetics(metadata, expected, as_collection):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        add_precursor_formula,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("precursor_formula") == expected


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_negative_element_counts_yield_warning_and_none(caplog, as_collection):
    spectrum_in = SpectrumBuilder().with_metadata({"formula": "H2O", "adduct": "[M-Na]+"}).build()

    with caplog.at_level(logging.WARNING):
        spectrum = run_filter_as_spectrum_or_collection(
            add_precursor_formula,
            spectrum_in,
            as_collection,
        )

    assert spectrum.get("precursor_formula") is None
    assert any("negative" in rec.message.lower() or "not set" in rec.message.lower() for rec in caplog.records)


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_zero_counts_removed_from_output(as_collection):
    spectrum_in = SpectrumBuilder().with_metadata({"formula": "CH4O", "adduct": "[M+Na-Na]+"}).build()

    spectrum = run_filter_as_spectrum_or_collection(
        add_precursor_formula,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("precursor_formula") == "CH4O"
    assert "Na" not in spectrum.get("precursor_formula")


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "formula, adduct, expected",
    [
        # Hill notation: C then H then alphabetical.
        ("CH4NO", "[M]+", "CH4NO"),
        # Scrambled input order should still output Hill order.
        ("O2H2C", "[M]+", "CH2O2"),
        # Ensure '1' is omitted.
        ("C1H4N1O1", "[M]+", "CH4NO"),
        # No carbon: then alphabetical only.
        ("H3N2O1S1", "[M]+", "H3N2OS"),
    ],
)
def test_hill_notation_and_ones_omission(formula, adduct, expected, as_collection):
    spectrum_in = SpectrumBuilder().with_metadata({"formula": formula, "adduct": adduct}).build()

    spectrum = run_filter_as_spectrum_or_collection(
        add_precursor_formula,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("precursor_formula") == expected


def test_add_precursor_formula_clone_true_does_not_modify_original_object_identity_or_metadata():
    metadata = {"formula": "CH4O", "adduct": "[M+H]+"}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = add_precursor_formula(spectrum_in, clone=True)

    assert spectrum is not spectrum_in
    assert spectrum_in.get("precursor_formula") is None
    assert spectrum.get("precursor_formula") == "CH5O"


def test_add_precursor_formula_clone_false_modifies_original_spectrum():
    metadata = {"formula": "CH4O", "adduct": "[M+H]+"}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = add_precursor_formula(spectrum_in, clone=False)

    assert spectrum is spectrum_in
    assert spectrum_in.get("precursor_formula") == "CH5O"


def test_add_precursor_formula_collection_updates_multiple_rows():
    spectra = [
        SpectrumBuilder().with_metadata({"formula": "C12H15ClN2O3", "adduct": "[M+H]+"}).build(),
        SpectrumBuilder().with_metadata({"formula": "H2O", "adduct": "[M+Na-Na]+"}).build(),
        SpectrumBuilder().with_metadata({"formula": "C2H6", "adduct": "[M-H2O]+"}).build(),
    ]
    collection = SpectraCollection(spectra)

    processed = add_precursor_formula(collection)

    assert isinstance(processed, SpectraCollection)
    assert len(processed) == 3
    assert processed.metadata.loc[0, "precursor_formula"] == "C12H16ClN2O3"
    assert processed.metadata.loc[1, "precursor_formula"] == "H2O"
    assert "precursor_formula" not in processed.metadata.columns or processed.metadata.loc[2, "precursor_formula"] != processed.metadata.loc[2, "precursor_formula"]


def test_add_precursor_formula_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder()
            .with_metadata({"formula": "CH4O", "adduct": "[M+H]+"})
            .build()
        ]
    )

    processed = add_precursor_formula(collection, clone=False)

    assert processed is collection
    assert collection.metadata.loc[0, "precursor_formula"] == "CH5O"
