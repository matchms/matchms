import pytest
import numpy as np
import pandas as pd
from matchms import Spectrum, SpectraCollection
from tests.builder_Spectrum import SpectrumBuilder


@pytest.fixture
def sample_spectra():
    mz1 = np.array([100.00003, 110.2, 200.581], dtype="float")
    intensities1 = np.array([0.51, 1.0, 0.011], dtype="float") # Sum 1,521
    metadata1 = {"compound_name": "A", "precursor_mz": 444.0, "charge": -1, "retention_time": 100}

    mz2 = np.array([111.00213, 180.2, 332.342], dtype="float")
    intensities2 = np.array([0.52, 1.0, 0.7], dtype="float") # Sum 2,22
    metadata2 = {"compound_name": "B", "precursor_mz": 565.0, "charge": -1, "retention_time": 200}

    mz3 = np.array([111.00213, 200.1, 200.213], dtype="float")
    intensities3 = np.array([0.52, 0.5, 1], dtype="float") # Sum 2,02
    metadata3 = {"compound_name": "C", "precursor_mz": 664.0, "charge": -1, "retention_time": 150}

    s1 = SpectrumBuilder().with_mz(mz1).with_intensities(intensities1).with_metadata(metadata1).build()
    s2 = SpectrumBuilder().with_mz(mz2).with_intensities(intensities2).with_metadata(metadata2).build()
    s3 = SpectrumBuilder().with_mz(mz3).with_intensities(intensities3).with_metadata(metadata3).build()

    return [s1, s2, s3]

@pytest.fixture
def collection(sample_spectra):
    return SpectraCollection(sample_spectra, bin_size=0.01)


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


def test_add_metadata_series(collection):
    scores = pd.Series([0.95, 0.88, 0.99], name="quality_score")
    collection.add_metadata(scores)

    assert "quality_score" in collection.metadata.columns
    assert collection.metadata.loc[2, "quality_score"] == 0.99


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


def test_dropna(sample_spectra):
    empty_spec = Spectrum(mz=np.array([]), intensities=np.array([]), metadata={"name": "empty"})
    col = SpectraCollection(sample_spectra + [empty_spec], bin_size=1.0)

    assert len(col) == 4
    clean_col = col.dropna()
    assert len(clean_col) == 3


def test_copy(collection):
    cloned = collection.copy()
    assert cloned is not collection
    assert np.array_equal(cloned.fragments.toarray(), collection.fragments.toarray())


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
