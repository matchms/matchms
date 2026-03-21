import numpy as np
import pandas as pd
import pytest
from matchms import SpectraCollection, Spectrum
from tests.builder_Spectrum import SpectrumBuilder


@pytest.fixture
def sample_spectra():
    mz1 = np.array([100.00003, 110.2, 200.581], dtype="float")
    intensities1 = np.array([0.51, 1.0, 0.011], dtype="float")  # Sum 1,521
    metadata1 = {"compound_name": "A", "precursor_mz": 444.0, "charge": -1, "retention_time": 100}

    mz2 = np.array([111.00213, 180.2, 332.342], dtype="float")
    intensities2 = np.array([0.52, 1.0, 0.7], dtype="float")  # Sum 2,22
    metadata2 = {"compound_name": "B", "precursor_mz": 565.0, "charge": -1, "retention_time": 200}

    mz3 = np.array([111.00213, 200.1, 200.213], dtype="float")
    intensities3 = np.array([0.52, 0.5, 1], dtype="float")  # Sum 2,02
    metadata3 = {"compound_name": "C", "precursor_mz": 664.0, "charge": -1, "retention_time": 150}

    s1 = SpectrumBuilder().with_mz(mz1).with_intensities(intensities1).with_metadata(metadata1).build()
    s2 = SpectrumBuilder().with_mz(mz2).with_intensities(intensities2).with_metadata(metadata2).build()
    s3 = SpectrumBuilder().with_mz(mz3).with_intensities(intensities3).with_metadata(metadata3).build()

    return [s1, s2, s3]


@pytest.fixture
def collection(sample_spectra):
    return SpectraCollection(sample_spectra, bin_size=0.01)


def test_getitem_slice(collection):
    sub_col = collection[0:2]

    assert isinstance(sub_col, SpectraCollection)
    assert len(sub_col) == 2
    assert sub_col.metadata["compound_name"].tolist() == ["A", "B"]

    original_sums = collection.fragments.sum(axis=1)
    sub_sums = sub_col.fragments.sum(axis=1)
    np.testing.assert_allclose(sub_sums, original_sums[0:2])


def test_getitem_list_indices(collection):
    indices = [0, 2]
    sub_col = collection[indices]

    assert len(sub_col) == 2
    assert sub_col.metadata["compound_name"].tolist() == ["A", "C"]

    np.testing.assert_allclose(sub_col.fragments.sum(axis=1), [1.521, 2.02], atol=1e-5)


def test_fragments_proxy_slicing(collection):
    raw_slice = collection.fragments[0:2]

    assert hasattr(raw_slice, "shape")
    assert raw_slice.shape[0] == 2
    assert not isinstance(raw_slice, SpectraCollection)


def test_fragments_mz_slicing(collection):
    sliced = collection.fragments[:, 100.0:150.0]

    assert sliced.shape[0] == 3
    assert sliced.shape[1] >= 1

    # Peak around 200 should be absent after slicing
    mz0, _ = sliced.get_row(0)
    assert np.all(mz0 < 150.1)


def test_metadata_extraction(collection):
    df = collection._metadata
    assert len(df) == 3
    assert list(df["compound_name"]) == ["A", "B", "C"]
    assert list(df["precursor_mz"]) == [444.0, 565.0, 664.0]
    assert list(df["retention_time"]) == [100, 200, 150]


def test_fragments_matrix_sums(collection):
    sums = collection.fragments.sum(axis=1)
    np.testing.assert_allclose(sums, [1.521, 2.22, 2.02], atol=1e-5)


def test_sort_by_rt_descending(collection):
    sorted_col = collection.sort(by="retention_time", on="metadata", ascending=False)

    assert sorted_col.metadata["compound_name"].tolist() == ["B", "C", "A"]
    assert sorted_col.metadata["retention_time"].tolist() == [200, 150, 100]

    sorted_sums = sorted_col.fragments.sum(axis=1)
    assert sorted_sums[0] == pytest.approx(2.22)
    assert sorted_sums[2] == 1.521


def test_sort_inplace(collection):
    original_id = id(collection)

    result = collection.sort(by="retention_time", on="metadata", ascending=False, inplace=True)

    assert result is None
    assert id(collection) == original_id
    assert collection.metadata["compound_name"].tolist() == ["B", "C", "A"]


def test_getitem_consistency(collection):
    spec_b = collection[1]

    assert spec_b.metadata["compound_name"] == "B"
    assert spec_b.metadata["precursor_mz"] == 565.0
    assert np.sum(spec_b.intensities) == pytest.approx(2.22)


def test_drop_spectra(collection):
    new_col = collection.drop(indices=[1])

    assert len(new_col) == 2
    assert new_col.metadata["compound_name"].tolist() == ["A", "C"]

    print(new_col[1].mz)

    assert np.any(np.isclose(new_col[1].mz, 200.105, atol=0.1))


def test_drop_duplicates(sample_spectra):
    col = SpectraCollection(sample_spectra + [sample_spectra[1]], bin_size=0.01)

    assert len(col) == 4
    deduped = col.drop_duplicates()

    assert len(deduped) == 3
    assert deduped.metadata["compound_name"].tolist() == ["A", "B", "C"]


def test_add_metadata_series(collection):
    scores = pd.Series([0.95, 0.88, 0.99], name="quality_score")
    collection.add_metadata(scores)

    assert "quality_score" in collection.metadata.columns
    assert collection.metadata.loc[2, "quality_score"] == 0.99


def test_add_metadata_overwrite(collection):
    new_rt = pd.Series([10, 20, 30], name="retention_time")
    collection.add_metadata(new_rt, overwrite=True)

    assert collection.metadata["retention_time"].tolist() == [10, 20, 30]


def test_sort_by_metadata(collection):
    # Sort by retention time (rt): [100, 200, 150] -> [100, 150, 200]
    sorted_col = collection.sort(by="retention_time", on="metadata", inplace=False)

    assert sorted_col.metadata["retention_time"].tolist() == [100, 150, 200]
    assert sorted_col.metadata["compound_name"].tolist() == ["A", "C", "B"]
    original_sums = collection.fragments.sum(axis=1)
    sorted_sums = sorted_col.fragments.sum(axis=1)
    assert sorted_sums[1] == original_sums[2]


def test_sort_by_fragments(collection):
    # Sort by intensity sums
    # A: 6.0, B: 12.0, C: 11.0 -> Ascending: A, C, B
    sorted_col = collection.sort(by="sum", on="fragments", ascending=True)
    assert sorted_col.metadata["compound_name"].tolist() == ["A", "C", "B"]


def test_drop_indices(collection):
    dropped = collection.drop([1])
    assert len(dropped) == 2
    assert dropped.metadata["compound_name"].tolist() == ["A", "C"]


def test_drop_inplace(collection):
    original_id = id(collection)

    result = collection.drop([1], inplace=True)

    assert result is None
    assert id(collection) == original_id
    assert len(collection) == 2
    assert collection.metadata["compound_name"].tolist() == ["A", "C"]


def test_dropna(sample_spectra):
    empty_spec = Spectrum(mz=np.array([]), intensities=np.array([]), metadata={"name": "empty"})
    col = SpectraCollection(sample_spectra + [empty_spec], bin_size=1.0)

    assert len(col) == 4
    clean_col = col.dropna()
    assert len(clean_col) == 3


def test_copy(collection):
    cloned = collection.copy()

    assert cloned is not collection
    assert cloned.metadata is not collection.metadata
    assert cloned.fragments is not collection.fragments

    np.testing.assert_array_equal(
        cloned._fragments.array.data,
        collection._fragments.array.data,
    )
    np.testing.assert_array_equal(
        cloned._fragments.array.indptr,
        collection._fragments.array.indptr,
    )
    np.testing.assert_array_equal(
        cloned._fragments.array.indices,
        collection._fragments.array.indices,
    )
    pd.testing.assert_frame_equal(cloned.metadata, collection.metadata)

    # Mutating the clone should not affect the original
    cloned.add_metadata(pd.Series([1, 2, 3], name="new_col"))

    assert "new_col" in cloned.metadata.columns
    assert "new_col" not in collection.metadata.columns


def test_mz_bin_conversion(collection):
    mz = 123.456
    bin_idx = collection.mz_to_bin(mz)
    back_mz = collection.bin_to_mz(bin_idx)

    assert bin_idx == 12345
    assert back_mz == 123.455


def test_describe(collection):
    stats = collection.describe()
    assert "peak_counts" in stats.columns
    assert "intensity_sums" in stats.columns
    assert stats.attrs["num_spectra"] == 3


def test_filter_by_metadata_mask(collection):
    mask = collection.metadata["retention_time"] > 120
    filtered = collection.filter(mask)

    # Should keep B (200) and C (150)
    assert len(filtered) == 2
    assert filtered.metadata["compound_name"].tolist() == ["B", "C"]
    assert filtered.metadata["retention_time"].tolist() == [200, 150]


def test_filter_by_fragments_mask(collection):
    mask = collection.fragments.sum(axis=1) > 2.0
    filtered = collection.filter(mask)

    assert len(filtered) == 2
    assert filtered.metadata["compound_name"].tolist() == ["B", "C"]


def test_filter_inplace(collection):
    original_id = id(collection)
    mask = np.array([True, False, True])  # Keep A and C

    result = collection.filter(mask, inplace=True)

    assert result is None
    assert id(collection) == original_id
    assert len(collection) == 2
    assert collection.metadata["compound_name"].tolist() == ["A", "C"]


def test_filter_invalid_length_raises_error(collection):
    short_mask = np.array([True, False])

    with pytest.raises(ValueError, match=r"Shape of filter mask \(2\) does not fit Items in SpectraCollection \(3\)."):
        collection.filter(short_mask)


def test_iteration_returns_spectrum_objects(collection):
    spectra = list(collection)

    assert len(spectra) == 3
    assert all(isinstance(s, Spectrum) for s in spectra)
    assert [s.metadata["compound_name"] for s in spectra] == ["A", "B", "C"]


def test_getitem_negative_row_slice(collection):
    sub_col = collection[-2:]

    assert isinstance(sub_col, SpectraCollection)
    assert len(sub_col) == 2
    assert sub_col.metadata["compound_name"].tolist() == ["B", "C"]
    np.testing.assert_allclose(sub_col.fragments.sum(axis=1), [2.22, 2.02], atol=1e-5)


def test_getitem_row_and_mz_slice_returns_collection(collection):
    sub_col = collection[:2, 100.0:150.0]

    assert isinstance(sub_col, SpectraCollection)
    assert len(sub_col) == 2
    assert sub_col.metadata["compound_name"].tolist() == ["A", "B"]

    mz0 = sub_col[0].mz
    mz1 = sub_col[1].mz

    assert np.all((mz0 >= 100.0) & (mz0 < 150.1))
    assert np.all((mz1 >= 100.0) & (mz1 < 150.1))

    assert np.sum(sub_col[0].intensities) == pytest.approx(1.51, abs=1e-6)
    assert np.sum(sub_col[1].intensities) == pytest.approx(0.52, abs=1e-6)


def test_getitem_scalar_row_and_mz_slice_returns_spectrum(collection):
    spec = collection[1, 100.0:200.0]

    assert isinstance(spec, Spectrum)
    assert spec.metadata["compound_name"] == "B"
    assert np.all((spec.mz >= 100.0) & (spec.mz < 200.1))
    assert np.sum(spec.intensities) == pytest.approx(1.52, abs=1e-6)


def test_getitem_row_mask_and_mz_slice(collection):
    mask = np.array([True, False, True])
    sub_col = collection[mask, 100.0:150.0]

    assert isinstance(sub_col, SpectraCollection)
    assert len(sub_col) == 2
    assert sub_col.metadata["compound_name"].tolist() == ["A", "C"]

    assert np.sum(sub_col[0].intensities) == pytest.approx(1.51, abs=1e-6)
    assert np.sum(sub_col[1].intensities) == pytest.approx(0.52, abs=1e-6)


def test_getitem_row_list_and_mz_slice(collection):
    sub_col = collection[[0, 2], 200.0:250.0]

    assert isinstance(sub_col, SpectraCollection)
    assert len(sub_col) == 2
    assert sub_col.metadata["compound_name"].tolist() == ["A", "C"]

    # A has one peak around 200.581, C has two around 200.1 and 200.213
    assert len(sub_col[0].mz) == 1
    assert len(sub_col[1].mz) == 2
    assert np.all(sub_col[0].mz >= 200.0)
    assert np.all(sub_col[1].mz >= 200.0)


def test_getitem_scalar_row_and_mz_slice_empty_result(collection):
    spec = collection[0, 300.0:400.0]

    assert isinstance(spec, Spectrum)
    assert spec.metadata["compound_name"] == "A"
    assert len(spec.mz) == 0
    assert len(spec.intensities) == 0


def test_getitem_tuple_invalid_length_raises(collection):
    with pytest.raises(IndexError, match="Expected at most two indexers"):
        _ = collection[0, 1, 2]


def test_getitem_invalid_row_mask_length_raises(collection):
    mask = np.array([True, False])

    with pytest.raises(ValueError, match=r"Shape of row selector \(2\) does not fit Items in SpectraCollection \(3\)."):
        _ = collection[mask, 100.0:200.0]


def test_iteration_after_2d_slice_returns_spectrum_objects(collection):
    sub_col = collection[:, 100.0:150.0]
    spectra = list(sub_col)

    assert len(spectra) == 3
    assert all(isinstance(s, Spectrum) for s in spectra)
    assert all(np.all((s.mz >= 100.0) & (s.mz < 150.1)) or len(s.mz) == 0 for s in spectra)
