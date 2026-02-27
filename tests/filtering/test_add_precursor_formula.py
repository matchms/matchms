import logging
import pytest
from matchms.filtering import add_precursor_formula
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, expected_formula",
    [
        [{"formula": "C12H15ClN2O3", "adduct": "[M+H]+"}, "C12H16ClN2O3"],
        [{"formula": "C12H15ClN2O3", "adduct": "[M+H-H2O]+"}, "C12H14ClN2O2"],  # multiple adducts
        [{"formula": "C12H15ClN2O3"}, None],  # no adduct
        [{"adduct": "[M+H-H2O]+"}, None],  # no formula
        [{"formula": "C2H8NO", "adduct": "[2M+H-H2O]+"}, "C4H15N2O"],  # multiple masses
        [{"formula": "C2H6", "adduct": "[M-H2O]+"}, None],  # impossible adduct formula combo
        [{}, None],
    ],
)
def test_derive_formula_from_smiles(metadata, expected_formula):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    spectrum = add_precursor_formula(spectrum_in)
    assert spectrum.get("precursor_formula") == expected_formula, "Expected different formula."


@pytest.mark.parametrize(
    "metadata, expected",
    [
        # Zero-cancellation: +Na -Na should yield the original formula with no Na present
        ({"formula": "H2O", "adduct": "[M+Na-Na]+"}, "H2O"),
        # Multiple ions with multiplicities: +2H +Na -3H2O
        # Base: C6H6 → +2H ⇒ C6H8; +Na ⇒ C6H8Na; -3H2O ⇒ C6H2NaO-? Wait:
        # subtracting 3 H2O removes H6O3, so C6H8 - H6 = C6H2, O0 + O3(-) = O-3 (negative) → impossible → expect None
        # Use a base that has enough O/H to subtract 3H2O safely:
        # Base: C6H12O3, +2H → C6H14O3, +Na → C6H14NaO3, -3H2O → C6H8Na  (O3-3=0, H14-6=8)
        ({"formula": "C6H12O3", "adduct": "[M+2H+Na-3H2O]+"}, "C6H8Na"),
        # Multiple parent masses: [3M+H]+
        # Base: C2H6O → 3M: C6H18O3, +H → C6H19O3
        ({"formula": "C2H6O", "adduct": "[3M+H]+"}, "C6H19O3"),
    ],
)
def test_various_adduct_arithmetics(metadata, expected):
    s = SpectrumBuilder().with_metadata(metadata).build()
    out = add_precursor_formula(s)
    assert out.get("precursor_formula") == expected


def test_negative_element_counts_yield_warning_and_none(caplog):
    # Subtract an element not present in the base: H2O with -Na
    s = SpectrumBuilder().with_metadata({"formula": "H2O", "adduct": "[M-Na]+"}).build()
    with caplog.at_level(logging.WARNING):
        out = add_precursor_formula(s)
    assert out.get("precursor_formula") is None
    # Ensure a helpful warning was logged
    assert any("negative" in rec.message.lower() or "not set" in rec.message.lower() for rec in caplog.records)


def test_zero_counts_removed_from_output():
    # Add and subtract the same ion multiplicity resulting in zero of that element
    # Base: CH4O; +Na -Na should return CH4O (no Na0)
    s = SpectrumBuilder().with_metadata({"formula": "CH4O", "adduct": "[M+Na-Na]+"}).build()
    out = add_precursor_formula(s)
    assert out.get("precursor_formula") == "CH4O"
    # sanity: ensure no "Na" appears
    assert "Na" not in out.get("precursor_formula")


@pytest.mark.parametrize(
    "formula, adduct, expected",
    [
        # Hill notation: C then H then alphabetical
        # Base already Hill: CH4N O → normalized to CH4NO
        ("CH4NO", "[M]+", "CH4NO"),
        # Scrambled input order should still output Hill order
        ("O2H2C", "[M]+", "CH2O2"),
        # Ensure '1' is omitted
        ("C1H4N1O1", "[M]+", "CH4NO"),
        # No carbon: then alphabetical only
        ("H3N2O1S1", "[M]+", "H3N2OS"),
    ],
)
def test_hill_notation_and_ones_omission(formula, adduct, expected):
    s = SpectrumBuilder().with_metadata({"formula": formula, "adduct": adduct}).build()
    out = add_precursor_formula(s)
    assert out.get("precursor_formula") == expected


def test_clone_true_does_not_modify_original_object_identity_or_metadata():
    metadata = {"formula": "CH4O", "adduct": "[M+H]+"}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    out = add_precursor_formula(spectrum_in, clone=True)

    # Ensure a new object was returned (clone)
    assert out is not spectrum_in

    # Original should remain without the derived field
    assert spectrum_in.get("precursor_formula") is None

    # New spectrum should have the field set
    assert out.get("precursor_formula") == "CH5O"
