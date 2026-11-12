import importlib
import logging
import numpy as np
import pandas as pd
import pytest
from matchms import SpectraCollection
from matchms.filtering.metadata_processing.require_precursor_mz import (
    _require_precursor_mz,
    require_precursor_mz,
)
from ..builder_spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{"precursor_mz": 60.0}, SpectrumBuilder().with_metadata({"precursor_mz": 60}).build()],
        [{"precursor_mz": 0.0}, None],
        [{"precursor_mz": -3.5}, None],
        [{}, None],
    ],
)
def test_require_precursor_mz_spectrum(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_precursor_mz(spectrum_in)

    assert spectrum == expected, "Expected no changes."


@pytest.mark.parametrize(
    "metadata, expected_kept",
    [
        [{"precursor_mz": 60.0}, True],
        [{"precursor_mz": 0.0}, False],
        [{"precursor_mz": -3.5}, False],
        [{}, False],
    ],
)
def test_require_precursor_mz_collection_single_row(metadata, expected_kept):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    collection = SpectraCollection([spectrum_in])

    filtered = require_precursor_mz(collection)

    assert isinstance(filtered, SpectraCollection)
    assert len(filtered) == int(expected_kept)


def test_require_precursor_mz_returns_same_spectrum_when_kept():
    """Requirement filters do not clone spectra; they only keep or remove them."""
    spectrum_in = SpectrumBuilder().with_metadata({"precursor_mz": 100.0}).build()

    spectrum = require_precursor_mz(spectrum_in)

    assert spectrum is spectrum_in


def test_require_precursor_mz_with_input_none():
    """Test if input spectrum is None."""
    spectrum = require_precursor_mz(None)

    assert spectrum is None


@pytest.mark.parametrize("precursor_mz", [0, 9.0, -200])
def test_require_precursor_mz_fail_when_mz_too_small(precursor_mz):
    """Test if spectrum is None when precursor_mz <= minimum_accepted_mz."""
    spectrum_in = SpectrumBuilder().build()
    spectrum_in.set("precursor_mz", precursor_mz)

    spectrum = require_precursor_mz(spectrum_in, minimum_accepted_mz=10)

    assert spectrum is None, "Expected spectrum to be None."


def test_require_precursor_mz_fail_because_below_zero():
    """Test if spectrum is None when precursor_mz < 0."""
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()
    spectrum_in.set("precursor_mz", -3.5)

    spectrum = require_precursor_mz(spectrum_in)

    assert spectrum is None, "Expected spectrum to be None."


def test_require_precursor_mz_collection_multiple_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"precursor_mz": 60.0}).build(),
            SpectrumBuilder().with_metadata({"precursor_mz": 0.0}).build(),
            SpectrumBuilder().with_metadata({"precursor_mz": -3.5}).build(),
            SpectrumBuilder().with_metadata({}).build(),
            SpectrumBuilder().with_metadata({"precursor_mz": 120.0}).build(),
        ]
    )

    filtered = require_precursor_mz(collection)

    assert filtered is not collection
    assert len(filtered) == 2
    assert filtered.metadata["precursor_mz"].tolist() == [60.0, 120.0]


def test_require_precursor_mz_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"precursor_mz": 60.0}).build(),
            SpectrumBuilder().with_metadata({"precursor_mz": 0.0}).build(),
        ]
    )

    filtered = require_precursor_mz(collection, clone=False)

    assert filtered is collection
    assert len(collection) == 1
    assert collection.metadata.loc[0, "precursor_mz"] == 60.0


# ---------------------------------------------------------------------------
# Explicit collection storage fixtures
# ---------------------------------------------------------------------------


def _collection_with_precursors(values, *, dtype=object):
    """Construct distinctive peak rows and force the requested metadata dtype.

    SpectraCollection normally harmonizes precursor_mz to a numeric dtype.
    Setting the column after construction is intentional: it lets us exercise
    the mixed/object fast path without changing production normalization code.
    """
    spectra = [
        SpectrumBuilder().with_mz(np.array([100.0 + i])).with_intensities(np.array([1.0])).build()
        for i in range(len(values))
    ]
    collection = SpectraCollection(spectra)
    collection._metadata["precursor_mz"] = pd.Series(values, dtype=dtype)
    assert collection.metadata["precursor_mz"].dtype == pd.Series(values, dtype=dtype).dtype
    return collection


def _collection_without_precursor(n, *, pepmass=None):
    collection = _collection_with_precursors([100.0] * n, dtype="float64")
    collection._metadata.drop(columns=["precursor_mz"], inplace=True)
    if pepmass is not None:
        collection._metadata["pepmass"] = pd.Series(pepmass, dtype=object)
    return collection


def _assert_kept_rows(filtered, original_row_ids):
    """Check peak-row alignment, not only the number of surviving rows."""
    assert len(filtered) == len(original_row_ids)
    for output_row, original_row in enumerate(original_row_ids):
        mz, intensities = filtered.fragments.get_row(output_row)
        np.testing.assert_allclose(mz, [100.0 + original_row], atol=1e-6)
        np.testing.assert_allclose(intensities, [1.0])


# ---------------------------------------------------------------------------
# Scalar semantics and bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "precursor,minimum,maximum,expected",
    [
        (10.0, 10.0, 20.0, True),
        (20.0, 10.0, 20.0, True),
        (9.99, 10.0, 20.0, False),
        (20.01, 10.0, 20.0, False),
        (np.int64(15), 10.0, 20.0, True),
        (np.float32(15), 10.0, 20.0, True),
        (None, 10.0, 20.0, False),
        (np.nan, 10.0, 20.0, False),
        (-5.0, None, None, True),
        (50.0, None, 20.0, False),
        (-5.0, None, 20.0, True),
        (10.0, 10.0, None, True),
    ],
)
def test_scalar_precursor_requirement_boundaries(precursor, minimum, maximum, expected):
    assert (
        _require_precursor_mz(
            {"precursor_mz": precursor},
            minimum_accepted_mz=minimum,
            maximum_mz=maximum,
        )
        is expected
    )


@pytest.mark.parametrize("value", ["15.0", "not-a-number", "invalid-mz"])
def test_scalar_string_precursor_raises_value_error(value):
    """Metadata conversion requires numeric scalars; it never parses strings."""
    with pytest.raises(ValueError, match="Expected scalar numeric metadata value"):
        _require_precursor_mz({"precursor_mz": value})


def test_scalar_missing_precursor_with_pepmass_raises_helpful_assertion():
    with pytest.raises(AssertionError, match="add_precursor_mz"):
        _require_precursor_mz({"pepmass": (250.0, 100.0)})


@pytest.mark.parametrize("pepmass", [None, [], "unknown", (None, None)])
def test_scalar_missing_precursor_without_usable_pepmass_is_rejected(pepmass):
    assert not _require_precursor_mz({"pepmass": pepmass})


# ---------------------------------------------------------------------------
# Vectorized fast path 1: genuinely numeric DataFrame columns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,values,kept",
    [
        ("float64", [np.nan, 9.0, 10.0, 15.0, 20.0, 21.0], [2, 3, 4]),
        ("Float64", [pd.NA, 9.0, 10.0, 15.0, 20.0, 21.0], [2, 3, 4]),
        ("Int64", [pd.NA, 9, 10, 15, 20, 21], [2, 3, 4]),
    ],
)
def test_collection_numeric_fast_path_inclusive_bounds_and_missing(dtype, values, kept):
    collection = _collection_with_precursors(values, dtype=dtype)
    original_row_count = len(collection)
    filtered = require_precursor_mz(collection, minimum_accepted_mz=10.0, maximum_mz=20.0)
    _assert_kept_rows(filtered, kept)
    assert filtered is not collection
    assert len(collection) == original_row_count


def test_collection_numeric_fast_path_without_bounds_keeps_negative_values():
    collection = _collection_with_precursors([-5.0, 0.0, 15.0, np.nan], dtype="float64")
    filtered = require_precursor_mz(collection, minimum_accepted_mz=None, maximum_mz=None)
    _assert_kept_rows(filtered, [0, 1, 2])


def test_collection_numeric_fast_path_reports_below_and_above_counts(caplog):
    collection = _collection_with_precursors([5.0, 7.0, 10.0, 20.0, 21.0, 30.0], dtype="float64")
    with caplog.at_level(logging.INFO, logger="matchms"):
        filtered = require_precursor_mz(collection, minimum_accepted_mz=10, maximum_mz=20)
    _assert_kept_rows(filtered, [2, 3])
    assert "2 spectra removed because precursor m/z was below" in caplog.text
    assert "2 spectra removed because precursor m/z was above" in caplog.text


# ---------------------------------------------------------------------------
# Vectorized fast path 2: object dtype, numeric scalars plus fallback records
# ---------------------------------------------------------------------------


def test_collection_object_fast_path_numeric_scalars_never_call_row_fallback(monkeypatch):
    values = [np.int64(12), np.float32(10), 20.0, 9.9, 20.1, np.nan, np.uint64(17), np.float64(20)]
    collection = _collection_with_precursors(values, dtype=object)
    assert collection.metadata["precursor_mz"].dtype == object

    module = importlib.import_module("matchms.filtering.metadata_processing.require_precursor_mz")

    def forbidden_fallback(*args, **kwargs):
        pytest.fail("Fast path 2 must not evaluate ordinary numeric precursor rows individually")

    monkeypatch.setattr(module, "_require_precursor_mz", forbidden_fallback)
    filtered = require_precursor_mz(collection, minimum_accepted_mz=10, maximum_mz=20)
    _assert_kept_rows(filtered, [0, 1, 2, 6, 7])


def test_collection_object_fast_path_falls_back_only_for_missing_values(monkeypatch):
    """Object dtype must use vectorization for numbers and scalar logic for missing entries."""
    values = [
        np.int64(12),
        np.float32(10),
        19.0,
        9.9,
        21.0,
        np.nan,
        np.float64(20),
        None,
        pd.NA,
    ]
    collection = _collection_with_precursors(values, dtype=object)
    module = importlib.import_module("matchms.filtering.metadata_processing.require_precursor_mz")
    actual_scalar = module._require_precursor_mz
    fallback_rows = []

    def tracked_scalar(metadata, minimum_accepted_mz=10.0, maximum_mz=None):
        fallback_rows.append(metadata.name)
        return actual_scalar(metadata, minimum_accepted_mz, maximum_mz)

    monkeypatch.setattr(module, "_require_precursor_mz", tracked_scalar)
    filtered = require_precursor_mz(collection, minimum_accepted_mz=10, maximum_mz=20)
    _assert_kept_rows(filtered, [0, 1, 2, 6])
    assert fallback_rows == [7, 8]
    assert len(collection) == len(values)  # default clone=True leaves source intact


@pytest.mark.parametrize("string_value", ["15.0", "not-a-number"])
def test_collection_object_fast_path_string_uses_scalar_fallback_and_raises(monkeypatch, string_value):
    """A string is not a numeric scalar, even if it can be parsed as a float."""
    collection = _collection_with_precursors([12, string_value, 20], dtype=object)
    module = importlib.import_module("matchms.filtering.metadata_processing.require_precursor_mz")
    actual_scalar = module._require_precursor_mz
    fallback_rows = []

    def tracked_scalar(metadata, minimum_accepted_mz=10.0, maximum_mz=None):
        fallback_rows.append(metadata.name)
        return actual_scalar(metadata, minimum_accepted_mz, maximum_mz)

    monkeypatch.setattr(module, "_require_precursor_mz", tracked_scalar)
    with pytest.raises(ValueError, match="Expected scalar numeric metadata value"):
        require_precursor_mz(collection, minimum_accepted_mz=10, maximum_mz=20)
    assert fallback_rows == [1]
    assert len(collection) == 3


def test_collection_object_fast_path_all_fallback_values():
    collection = _collection_with_precursors([None, pd.NA], dtype=object)
    filtered = require_precursor_mz(collection, minimum_accepted_mz=10, maximum_mz=20)
    _assert_kept_rows(filtered, [])


def test_collection_object_fast_path_with_no_bounds():
    collection = _collection_with_precursors([-4, 0.0, np.float32(15.5), None, np.nan, pd.NA], dtype=object)
    filtered = require_precursor_mz(collection, minimum_accepted_mz=None, maximum_mz=None)
    _assert_kept_rows(filtered, [0, 1, 2])


# ---------------------------------------------------------------------------
# Missing-column fallbacks, empty collections and cloning
# ---------------------------------------------------------------------------


def test_collection_missing_precursor_column_discards_all_rows():
    collection = _collection_without_precursor(3)
    assert "precursor_mz" not in collection.metadata.columns
    filtered = require_precursor_mz(collection)
    assert len(filtered) == 0
    assert len(collection) == 3


def test_collection_missing_precursor_column_checks_pepmass():
    collection = _collection_without_precursor(1, pepmass=[(250.0, 100.0)])
    with pytest.raises(AssertionError, match="add_precursor_mz"):
        require_precursor_mz(collection)


def test_collection_object_fallback_missing_precursor_checks_pepmass():
    collection = _collection_with_precursors([None, 20.0], dtype=object)
    collection._metadata["pepmass"] = pd.Series([(250.0, 100.0), None], dtype=object)
    with pytest.raises(AssertionError, match="add_precursor_mz"):
        require_precursor_mz(collection)


def test_collection_missing_precursor_column_with_unusable_pepmass_discards_rows():
    collection = _collection_without_precursor(3, pepmass=[None, [], "unknown"])
    filtered = require_precursor_mz(collection)
    assert len(filtered) == 0


@pytest.mark.parametrize("clone", [True, False, None])
def test_collection_clone_semantics_with_mixed_precursor_column(clone):
    collection = _collection_with_precursors([100, pd.NA, 200], dtype=object)
    original_mz = [collection.fragments.get_row(i)[0].copy() for i in range(3)]
    result = require_precursor_mz(collection, clone=clone)
    _assert_kept_rows(result, [0, 2])
    if clone:
        assert result is not collection
        assert len(collection) == 3
        for row, mz in enumerate(original_mz):
            np.testing.assert_array_equal(collection.fragments.get_row(row)[0], mz)
    else:
        assert result is collection
        assert len(collection) == 2


@pytest.mark.parametrize("clone", [True, False])
def test_empty_collection_fast_return_preserves_clone_semantics(clone):
    collection = _collection_with_precursors([0.0], dtype="float64")
    empty = require_precursor_mz(collection)
    assert len(empty) == 0
    returned = require_precursor_mz(empty, clone=clone)
    assert len(returned) == 0
    assert (returned is empty) is not clone
