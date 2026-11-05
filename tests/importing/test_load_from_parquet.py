from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from matchms import SpectraCollection, Spectrum
from matchms.importing import load_from_parquet


def test_load_from_parquet_using_filepath():
    module_root = Path(__file__).resolve().parents[1]
    spectra_file = module_root / "testdata" / "pesticides.parquet"

    spectra = load_from_parquet(str(spectra_file))

    assert len(spectra) > 0
    assert isinstance(spectra, SpectraCollection)
    assert isinstance(spectra[0], Spectrum)

    spectra = load_from_parquet(spectra_file)

    assert len(spectra) > 0
    assert isinstance(spectra, SpectraCollection)
    assert isinstance(spectra[0], Spectrum)


def test_load_missing_parquet_raises():
    with pytest.raises(FileNotFoundError):
        load_from_parquet("does-not-exist.parquet")


def test_load_kaggle_style_columns_and_filters(tmp_path):
    parquet_file = tmp_path / "kaggle_style.parquet"
    table = pa.table(
        {
            "ingest_lib": ["library-a", "enveda-180"],
            "inchikey14": ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"],
            "normalized_smiles": ["CCO", "CCN"],
            "precursor_mz": [101.1, 202.2],
            "ms2_mzs": pa.array(
                [[50.0, 75.0], [60.0, 80.0]],
                type=pa.list_(pa.float64()),
            ),
            "ms2_normalized_intensities": pa.array(
                [[1.0, 0.25], [1.0, 0.5]],
                type=pa.list_(pa.float32()),
            ),
        }
    )
    pq.write_table(table, parquet_file)

    spectra = load_from_parquet(
        parquet_file,
        columns=[
            "ingest_lib",
            "inchikey14",
            "normalized_smiles",
            "precursor_mz",
        ],
        filters=[("ingest_lib", "!=", "enveda-180")],
    )

    assert len(spectra) == 1
    assert list(spectra.metadata.columns) == [
        "ingest_lib",
        "inchikey14",
        "normalized_smiles",
        "precursor_mz",
    ]
    assert spectra[0].get("ingest_lib") == "library-a"
    np.testing.assert_allclose(spectra[0].mz, np.array([50.0, 75.0]))
    np.testing.assert_allclose(spectra[0].intensities, np.array([1.0, 0.25]))


def test_load_from_parquet_rejects_mismatched_peak_lengths(tmp_path):
    parquet_file = tmp_path / "bad.parquet"
    table = pa.table(
        {
            "mz": pa.array([[10.0, 20.0]], type=pa.list_(pa.float64())),
            "intensities": pa.array([[1.0]], type=pa.list_(pa.float64())),
        }
    )
    pq.write_table(table, parquet_file)

    with pytest.raises(ValueError, match="list lengths must match"):
        load_from_parquet(parquet_file)


def test_load_from_parquet_harmonizes_metadata_columns(tmp_path):
    parquet_file = tmp_path / "metadata.parquet"
    table = pa.table(
        {
            "Precursor_MZ": [123.45],
            "mz": pa.array([[10.0]], type=pa.list_(pa.float64())),
            "intensities": pa.array([[1.0]], type=pa.list_(pa.float64())),
        }
    )
    pq.write_table(table, parquet_file)

    spectra = load_from_parquet(parquet_file)

    assert "precursor_mz" in spectra.metadata.columns
    assert spectra[0].get("precursor_mz") == pytest.approx(123.45)


def test_pesticides_parquet_matches_mgf_fixture():
    from matchms import SpectraCollection
    from matchms.importing import load_from_mgf

    module_root = Path(__file__).resolve().parents[1]
    parquet_file = module_root / "testdata" / "pesticides.parquet"
    mgf_file = module_root / "testdata" / "pesticides.mgf"

    from_parquet = load_from_parquet(parquet_file)
    from_mgf = SpectraCollection(load_from_mgf(mgf_file))

    assert len(from_parquet) == len(from_mgf)
    for i in range(len(from_mgf)):
        np.testing.assert_allclose(from_parquet[i].mz, from_mgf[i].mz)
        np.testing.assert_allclose(from_parquet[i].intensities, from_mgf[i].intensities)
