"""Tests for the persistent SpectraCollection-native FlashIndex."""

import json
from types import SimpleNamespace
import numpy as np
import pytest
from matchms.similarity._flash_entropy import entropy_rows
from matchms.similarity._flash_prepared import empty_prepared, pack_native
from matchms.similarity.flash_index import FlashIndex, build_entropy_index, config_from_settings


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


def settings(dtype=np.float64):
    return ("entropy", 1., False, -1.6, 0., True, 0., np.dtype(dtype).str)


def prepared(dtype=np.float64):
    return pack_native(SimpleNamespace(
        spec_offsets=np.array([0, 3, 3, 5], np.int64),
        spec_mz=np.array([100., 100., 200., 100., 210.], dtype),
        spec_int=np.array([.1, .15, .25, .2, .3], dtype),
        precursor_mz=np.array([500., np.nan, 510.]),
        spec_l2=None,
    ), dtype, settings(dtype))


def make_index(mode="hybrid", dtype=np.float64):
    p = prepared(dtype)
    return build_entropy_index(p, mode, config=config_from_settings(p.settings),
                               metadata={"mz_precision": 1e-6, "custom": "kept"})


def downgrade_to_v2(source, target):
    with np.load(source, allow_pickle=False) as f:
        payload = {name: f[name] for name in f.files}
    meta = json.loads(str(payload["__metadata__"].item()))
    meta["version"] = 2
    for name in ("peaks_pid", "peaks_xlog2", "nl_xlog2"):
        payload.pop(name, None)
        meta["optional_arrays"].pop(name, None)
    payload["__metadata__"] = np.asarray(json.dumps(meta))
    np.savez(target, **payload)


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


@pytest.mark.parametrize("mode", ["fragment", "neutral_loss", "hybrid"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_new_public_archive_roundtrip_and_legacy_upgrade(tmp_path, mode, dtype):
    index = make_index(mode, dtype)
    file = tmp_path/"new.npz"
    index.save(file)
    with np.load(file, allow_pickle=False) as f:
        meta = json.loads(str(f["__metadata__"].item()))
        assert meta["format"] == "matchms.flash_index"
        assert meta["version"] == 3
        assert "tolerance" not in meta["config"]
        assert "matching_mode" not in meta["config"]
    old_file = tmp_path/"old.npz"
    downgrade_to_v2(file, old_file)
    old = FlashIndex.load(old_file)
    assert old.peaks_pid is None
    olddata = old.entropy_data()
    newdata = FlashIndex.load(file).entropy_data()
    for group in ("fragment", "neutral_loss"):
        for a, b in zip(getattr(newdata, group), getattr(olddata, group), strict=True):
            np.testing.assert_array_equal(a, b)
            assert not b.flags.writeable
    assert old.config == index.config and old.metadata == index.metadata
    old.save(tmp_path/"resaved.npz")
    again = FlashIndex.load(tmp_path/"resaved.npz")
    np.testing.assert_array_equal(again.peaks_pid, old.peaks_pid)


def test_legacy_duplicate_order_is_recovered_from_intensities():
    original = make_index()
    # Reverse the two equal-m/z records of spectrum 0 in the global view, as
    # an older unstable argsort could. No physical mapping supplied.
    perm = np.array([1, 0, 2, 3, 4])
    inverse = np.argsort(perm)
    values = original.__dict__.copy()
    for name in ("peaks_mz", "peaks_int", "peaks_spec_idx"):
        values[name] = values[name][perm]
    values.update(peaks_pid=None, peaks_xlog2=None, nl_xlog2=None,
                  nl_product_idx=inverse[original.nl_product_idx])
    old = FlashIndex(**values)
    for a, b in zip(old.entropy_data().fragment, original.entropy_data().fragment, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("mode", ["fragment", "neutral_loss", "hybrid"])
def test_legacy_loaded_entropy_scores_equal_fresh_index(tmp_path, mode):
    index = make_index(mode)
    file = tmp_path/"new.npz"
    oldfile = tmp_path/"old.npz"
    index.save(file)
    downgrade_to_v2(file, oldfile)
    outputs = []
    p = prepared()
    for source in (index, FlashIndex.load(oldfile)):
        data = source.entropy_data()
        out = np.empty((p.n_specs, source.n_specs))
        entropy_rows(out, p.spec_offsets, p.spec_mz, p.spec_int, p.precursor_mz,
                     data.fragment, data.neutral_loss, data.precursor_mz, data.n_peaks,
                     .02, False, {"fragment":0,"neutral_loss":1,"hybrid":2}[mode],
                     -1., False, 0, p.n_specs)
        outputs.append(out)
    np.testing.assert_array_equal(*outputs)


def test_kernel_cache_is_not_rebuilt_or_serialized(tmp_path, monkeypatch):
    index = make_index()
    data = index.entropy_data()
    def fail():
        raise AssertionError("Repeated physical-ID construction")
    monkeypatch.setattr(FlashIndex, "_physical_peak_ids", lambda _: fail())
    assert index.entropy_data() is data
    index.save(tmp_path/"index.npz")
    with np.load(tmp_path/"index.npz", allow_pickle=False) as f:
        assert "_runtime_cache" not in f.files


def test_cosine_casts_only_once_and_does_not_modify_caller():
    p = prepared(np.float32)
    source = make_index(dtype=np.float32)
    source.spec_l2 = np.array([.3, 0., .4], np.float32)
    source.peaks_spec_idx = source.peaks_spec_idx.astype(np.int32)
    data = source.cosine_data()
    assert source.cosine_data() is data
    assert data.peaks_spec_idx.dtype == np.int64
    assert data.spec_l2.dtype == np.float64
    assert source.peaks_spec_idx.dtype == np.int32
    assert source.spec_l2.dtype == np.float32
    assert source.peaks_int.flags.writeable
    assert not data.peaks_int.flags.writeable


@pytest.mark.parametrize("name, value, message", [
    ("peaks_pid", np.array([0,0,2,3,4]), "permutation"),
    ("peaks_pid", np.array([1,0,3,2,4]), "consistently"),
    ("peaks_pid", np.arange(5, dtype=float), "integer"),
    ("peaks_xlog2", np.ones(5), "disagrees"),
    ("spec_offsets", np.array([0.,3.,3.,5.]), "integers"),
    ("peaks_spec_idx", np.array([0.,0.,2.,0.,2.]), "integer"),
    ("nl_xlog2", np.ones(5), "disagrees"),
])
def test_reject_unsafe_postings_before_native_scoring(name, value, message):
    values = make_index().__dict__.copy()
    values[name] = value
    with pytest.raises(ValueError, match=message):
        FlashIndex(**values)


def test_save_preserves_overwrite_policy_and_can_opt_out(tmp_path):
    index = make_index()
    filename = tmp_path/"a"/"b.npz"
    index.save(filename)
    index.save(filename)  # The original public FlashIndex overwrites by default.
    with pytest.raises(FileExistsError):
        index.save(filename, overwrite=False)
    assert not list(filename.parent.glob("*.tmp"))


def test_empty_index_and_prepared_slices():
    empty = empty_prepared(np.float64, settings())
    index = build_entropy_index(empty, "hybrid", config=config_from_settings(empty.settings))
    assert index.n_specs == index.n_peaks == 0
    assert index.has_neutral_loss_index
    assert index.entropy_data().neutral_loss[0].size == 0
    p = prepared()
    assert p.slice(1,2).n_peaks == 0
    np.testing.assert_array_equal(p.slice(2,3).spec_offsets, [0,2])
    with pytest.raises(ValueError):
        p.slice(0,100)


def test_bad_required_or_optional_archive(tmp_path):
    index = make_index()
    file = tmp_path/"orig.npz"
    index.save(file)
    with np.load(file, allow_pickle=False) as f:
        arrays = {key:f[key] for key in f.files if key != "spec_offsets"}
    np.savez(tmp_path/"bad.npz", **arrays)
    with pytest.raises(ValueError, match="missing required"):
        FlashIndex.load(tmp_path/"bad.npz")


def test_inconsistent_legacy_views_are_not_silently_repaired():
    values = make_index("fragment").__dict__.copy()
    values.update(peaks_pid=None, peaks_xlog2=None)
    values["peaks_int"] = values["peaks_int"].copy()
    values["peaks_int"][0] += .01
    old = FlashIndex(**values)
    with pytest.raises(ValueError, match="inconsistent"):
        old.entropy_data()


def test_prepared_rejects_bad_shapes_nonfinite_or_unsorted():
    base = SimpleNamespace(spec_offsets=np.array([0,2]), spec_mz=np.array([200.,100.]),
                           spec_int=np.array([.25,.25]), precursor_mz=np.array([500.]))
    with pytest.raises(ValueError, match="sorted"):
        pack_native(base,np.float64,settings())
    base.spec_mz=np.array([100.,200.]);base.spec_int=np.array([.5,np.nan])
    with pytest.raises(ValueError, match="finite"):
        pack_native(base,np.float64,settings())
