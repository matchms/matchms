"""Tests for the persistent SpectraCollection-native FlashIndex."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from matchms.similarity.flash_index import FlashIndex


def _make_index(*, with_l2=True, with_neutral_loss=False):
    """Return a small but structurally realistic FlashIndex."""
    # Spectrum-major representation:
    # spectrum 0 -> (100, 0.2), (150, 0.8)
    # spectrum 1 -> (110, 0.3), (200, 0.7)
    spec_offsets = np.array([0, 2, 4], dtype=np.int64)
    spec_mz = np.array([100.0, 150.0, 110.0, 200.0], dtype=np.float64)
    spec_int = np.array([0.2, 0.8, 0.3, 0.7], dtype=np.float64)

    # Same peaks, globally sorted by product m/z.
    order = np.argsort(spec_mz)
    spec_idx = np.array([0, 0, 1, 1], dtype=np.int32)
    peaks_mz = spec_mz[order]
    peaks_int = spec_int[order]
    peaks_spec_idx = spec_idx[order]

    kwargs = {}
    if with_l2:
        kwargs["spec_l2"] = np.array(
            [np.hypot(0.2, 0.8), np.hypot(0.3, 0.7)],
            dtype=np.float64,
        )

    if with_neutral_loss:
        precursor_mz = np.array([250.0, 300.0], dtype=np.float64)
        neutral_loss = precursor_mz[spec_idx] - spec_mz
        nl_order = np.argsort(neutral_loss)

        # Map spectrum-major peak positions to the globally sorted product view.
        product_pos = np.empty(spec_mz.size, dtype=np.int64)
        product_pos[order] = np.arange(spec_mz.size, dtype=np.int64)

        kwargs.update(
            nl_mz=neutral_loss[nl_order],
            nl_int=spec_int[nl_order],
            nl_spec_idx=spec_idx[nl_order],
            nl_product_idx=product_pos[nl_order],
        )
    else:
        precursor_mz = np.array([250.0, np.nan], dtype=np.float64)

    return FlashIndex(
        n_specs=2,
        dtype=np.float64,
        peaks_mz=peaks_mz,
        peaks_int=peaks_int,
        peaks_spec_idx=peaks_spec_idx,
        spec_offsets=spec_offsets,
        spec_mz=spec_mz,
        spec_int=spec_int,
        precursor_mz=precursor_mz,
        config={
            "weighing_type": "cosine",
            "noise_cutoff": 0.01,
            "remove_precursor": True,
        },
        metadata={
            "mz_precision": 1e-6,
            "fragment_backend": "CSRFragmentCollection",
        },
        **kwargs,
    )


def _write_npz_with_metadata(path, metadata, **arrays):
    payload = {"__metadata__": np.asarray(json.dumps(metadata)), **arrays}
    with path.open("wb") as handle:
        np.savez(handle, **payload)


def test_flash_index_properties_and_repr():
    index = _make_index(with_l2=True, with_neutral_loss=True)

    assert index.has_l2_norms is True
    assert index.has_neutral_loss_index is True
    assert "n_specs=2" in repr(index)
    assert "n_peaks=4" in repr(index)
    assert "neutral_loss=True" in repr(index)
    assert "l2_norms=True" in repr(index)


def test_from_library_wraps_arrays_without_copying():
    source = _make_index(with_l2=True, with_neutral_loss=True)
    library = SimpleNamespace(
        n_specs=source.n_specs,
        dtype=source.dtype,
        peaks_mz=source.peaks_mz,
        peaks_int=source.peaks_int,
        peaks_spec_idx=source.peaks_spec_idx,
        spec_offsets=source.spec_offsets,
        spec_mz=source.spec_mz,
        spec_int=source.spec_int,
        precursor_mz=source.precursor_mz,
        spec_l2=source.spec_l2,
        nl_mz=source.nl_mz,
        nl_int=source.nl_int,
        nl_spec_idx=source.nl_spec_idx,
        nl_product_idx=source.nl_product_idx,
    )

    config = {"weighing_type": "cosine"}
    metadata = {"mz_precision": 1e-6}
    index = FlashIndex.from_library(library, config=config, metadata=metadata)

    assert index.peaks_mz is library.peaks_mz
    assert index.spec_mz is library.spec_mz
    assert index.spec_l2 is library.spec_l2
    assert index.nl_mz is library.nl_mz
    assert index.config == config
    assert index.metadata == metadata

    # The dictionaries themselves should not be shared with the caller.
    config["new"] = "value"
    metadata["new"] = "value"
    assert "new" not in index.config
    assert "new" not in index.metadata


@pytest.mark.parametrize(
    "with_l2, with_neutral_loss",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_save_load_roundtrip(tmp_path, with_l2, with_neutral_loss):
    index = _make_index(
        with_l2=with_l2,
        with_neutral_loss=with_neutral_loss,
    )
    filename = tmp_path / "library.flash.npz"

    index.save(filename)
    loaded = FlashIndex.load(filename)

    assert loaded.n_specs == index.n_specs
    assert loaded.dtype == index.dtype
    assert loaded.config == index.config
    assert loaded.metadata == index.metadata
    assert loaded.has_l2_norms is with_l2
    assert loaded.has_neutral_loss_index is with_neutral_loss

    required_arrays = (
        "peaks_mz",
        "peaks_int",
        "peaks_spec_idx",
        "spec_offsets",
        "spec_mz",
        "spec_int",
        "precursor_mz",
    )
    for name in required_arrays:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(index, name))

    optional_arrays = (
        "spec_l2",
        "nl_mz",
        "nl_int",
        "nl_spec_idx",
        "nl_product_idx",
    )
    for name in optional_arrays:
        expected = getattr(index, name)
        actual = getattr(loaded, name)
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_array_equal(actual, expected)

    assert not (tmp_path / "library.flash.npz.tmp").exists()


def test_save_creates_parent_directories(tmp_path):
    index = _make_index()
    filename = tmp_path / "nested" / "index" / "library.npz"

    index.save(filename)

    assert filename.is_file()


@pytest.mark.parametrize(
    "field,replacement,match",
    [
        ("spec_offsets", np.array([0, 4], dtype=np.int64), r"n_specs \+ 1"),
        ("spec_offsets", np.array([1, 2, 4], dtype=np.int64), "start at 0"),
        ("spec_offsets", np.array([0, 3, 2], dtype=np.int64), "monotonically"),
        ("spec_int", np.array([0.2, 0.8, 0.3]), "identical lengths"),
        ("precursor_mz", np.array([250.0]), "length n_specs"),
        ("peaks_mz", np.array([100.0, 200.0, 110.0, 150.0]), "globally sorted"),
        ("peaks_spec_idx", np.array([0, 1, 0, 2]), "out-of-range"),
        ("spec_l2", np.array([1.0]), "length n_specs"),
    ],
)
def test_structural_validation_rejects_invalid_arrays(field, replacement, match):
    index = _make_index(with_l2=True)
    values = index.__dict__.copy()
    values[field] = replacement

    with pytest.raises(ValueError, match=match):
        FlashIndex(**values)


def test_required_arrays_must_be_numpy_arrays():
    index = _make_index()
    values = index.__dict__.copy()
    values["spec_mz"] = [100.0, 150.0, 110.0, 200.0]

    with pytest.raises(TypeError, match="spec_mz must be a NumPy array"):
        FlashIndex(**values)


def test_neutral_loss_arrays_must_be_all_present_or_all_missing():
    index = _make_index(with_neutral_loss=True)
    values = index.__dict__.copy()
    values["nl_int"] = None

    with pytest.raises(ValueError, match="either all be present or all be None"):
        FlashIndex(**values)


def test_neutral_loss_product_positions_are_validated():
    index = _make_index(with_neutral_loss=True)
    values = index.__dict__.copy()
    values["nl_product_idx"] = index.nl_product_idx.copy()
    values["nl_product_idx"][0] = index.peaks_mz.size

    with pytest.raises(ValueError, match="out-of-range product positions"):
        FlashIndex(**values)


def test_load_rejects_wrong_format(tmp_path):
    index = _make_index()
    filename = tmp_path / "wrong_format.npz"
    index.save(filename)

    with np.load(filename, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"].item()))
        arrays = {name: np.asarray(archive[name]) for name in archive.files if name != "__metadata__"}

    metadata["format"] = "something.else"
    _write_npz_with_metadata(filename, metadata, **arrays)

    with pytest.raises(ValueError, match="Not a matchms Flash index"):
        FlashIndex.load(filename)


def test_load_rejects_unsupported_version(tmp_path):
    index = _make_index()
    filename = tmp_path / "wrong_version.npz"
    index.save(filename)

    with np.load(filename, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"].item()))
        arrays = {name: np.asarray(archive[name]) for name in archive.files if name != "__metadata__"}

    metadata["version"] += 1
    _write_npz_with_metadata(filename, metadata, **arrays)

    with pytest.raises(ValueError, match="Unsupported Flash index version"):
        FlashIndex.load(filename)


def test_load_rejects_missing_declared_optional_array(tmp_path):
    index = _make_index(with_l2=True)
    filename = tmp_path / "missing_optional.npz"
    index.save(filename)

    with np.load(filename, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"].item()))
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name not in {"__metadata__", "spec_l2"}
        }

    assert metadata["optional_arrays"]["spec_l2"] is True
    _write_npz_with_metadata(filename, metadata, **arrays)

    with pytest.raises(ValueError, match="declares array 'spec_l2'"):
        FlashIndex.load(filename)
