import numpy as np
import pyarrow.parquet as pq
import pytest
from matchms import SpectraCollection, Spectrum
from matchms.exporting import save_as_parquet
from matchms.importing import load_from_parquet


def _collection() -> SpectraCollection:
    spectra = [
        Spectrum(
            mz=np.array([10.12344, 20.0], dtype=float),
            intensities=np.array([1.0, 0.25], dtype=float),
            metadata={"id": "a", "precursor_mz": 100.5, "tags": np.array([1, 2])},
        ),
        Spectrum(
            mz=np.array([15.0, 30.0], dtype=float),
            intensities=np.array([0.5, 1.0], dtype=float),
            metadata={"id": "b", "precursor_mz": 200.5, "tags": np.array([3, 4])},
        ),
    ]
    return SpectraCollection(spectra, mz_precision=1e-4)


def test_save_as_parquet_roundtrip(tmp_path):
    parquet_file = tmp_path / "spectra.parquet"
    original = _collection()

    save_as_parquet(original, parquet_file)
    loaded = load_from_parquet(parquet_file)

    assert len(loaded) == len(original)
    assert loaded.mz_precision == pytest.approx(original.mz_precision)
    assert loaded.fragments.mz_precision == pytest.approx(original.fragments.mz_precision)
    assert loaded.fragments.mz_rounding == original.fragments.mz_rounding
    assert list(loaded.metadata["id"]) == ["a", "b"]

    for i in range(len(original)):
        np.testing.assert_allclose(loaded[i].mz, original[i].mz)
        np.testing.assert_allclose(loaded[i].intensities, original[i].intensities)


def test_save_as_parquet_writes_matchms_schema_metadata(tmp_path):
    parquet_file = tmp_path / "spectra.parquet"
    save_as_parquet(
        _collection(),
        parquet_file,
        mz_column="ms2_mzs",
        intensity_column="ms2_normalized_intensities",
    )

    schema = pq.read_schema(parquet_file)
    metadata = schema.metadata

    assert metadata[b"matchms_format"] == b"matchms.spectra_collection"
    assert metadata[b"matchms_format_version"] == b"1"
    assert metadata[b"matchms_mz_column"] == b"ms2_mzs"
    assert metadata[b"matchms_intensity_column"] == b"ms2_normalized_intensities"
    assert "ms2_mzs" in schema.names
    assert "ms2_normalized_intensities" in schema.names


def test_save_as_parquet_rejects_peak_metadata_collision(tmp_path):
    spectrum = Spectrum(
        mz=np.array([10.0], dtype=float),
        intensities=np.array([1.0], dtype=float),
        metadata={"mz": "metadata-value"},
    )

    with pytest.raises(ValueError, match="collide with metadata"):
        save_as_parquet([spectrum], tmp_path / "collision.parquet")


def test_save_as_parquet_can_write_empty_file(tmp_path):
    parquet_file = tmp_path / "empty.parquet"

    save_as_parquet([], parquet_file)

    table = pq.read_table(parquet_file)
    assert table.num_rows == 0
    assert table.column_names == ["mz", "intensities"]


def test_spectra_collection_to_parquet(tmp_path):
    parquet_file = tmp_path / "collection.parquet"
    collection = _collection()

    collection.to_parquet(parquet_file)
    loaded = load_from_parquet(parquet_file)

    assert len(loaded) == len(collection)

    with pytest.raises(FileExistsError):
        collection.to_parquet(parquet_file)
