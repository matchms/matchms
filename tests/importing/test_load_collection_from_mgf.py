from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from matchms import SpectraCollection
from matchms.fragment_collection import CSRFragmentCollection
from matchms.importing import load_collection_from_mgf, load_from_mgf, load_ms2_dataset
from matchms.metadata import Metadata, harmonize_metadata_values
from ..builder_spectrum import SpectrumBuilder


TESTDATA = Path(__file__).resolve().parents[1] / "testdata"
MGF_FIXTURES = [
    "pesticides.mgf",
    "testdata.mgf",
    "save_as_msp_from_mgf.mgf",
    "test_remove_fingerprint.mgf",
]


def _assert_same_collection(new: SpectraCollection, old: SpectraCollection):
    new_metadata = pd.DataFrame(new.metadata)
    old_metadata = pd.DataFrame(old.metadata)
    assert list(new_metadata.columns) == list(old_metadata.columns)
    pd.testing.assert_frame_equal(new_metadata, old_metadata)

    new_array, old_array = new.fragments.array, old.fragments.array
    assert new_array.shape == old_array.shape
    np.testing.assert_array_equal(new_array.indptr, old_array.indptr)
    np.testing.assert_array_equal(new_array.indices, old_array.indices)
    np.testing.assert_allclose(new_array.data, old_array.data)
    assert new.mz_precision == old.mz_precision
    assert new.fragments.mz_precision == old.fragments.mz_precision
    assert new.fragments.mz_rounding == old.fragments.mz_rounding


@pytest.mark.parametrize("filename", MGF_FIXTURES)
@pytest.mark.parametrize("metadata_harmonization", [True, False])
def test_load_collection_from_mgf_matches_spectrum_based_import(filename, metadata_harmonization):
    path = TESTDATA / filename
    expected = SpectraCollection(load_from_mgf(path, metadata_harmonization=metadata_harmonization))
    collection = load_collection_from_mgf(path, metadata_harmonization=metadata_harmonization)

    assert isinstance(collection, SpectraCollection)
    _assert_same_collection(collection, expected)


def test_load_collection_from_mgf_rows_reconstruct_the_same_spectra():
    path = TESTDATA / "pesticides.mgf"
    spectra = list(load_from_mgf(path))
    collection = load_collection_from_mgf(path)

    assert len(collection) == len(spectra)
    for row, spectrum in zip(range(len(collection)), spectra, strict=True):
        reconstructed = collection[row]
        np.testing.assert_allclose(reconstructed.mz, spectrum.mz, atol=1e-6)
        np.testing.assert_allclose(reconstructed.intensities, spectrum.intensities)
        assert reconstructed.get("precursor_mz") == pytest.approx(spectrum.get("precursor_mz"))
        assert reconstructed.get("compound_name") == spectrum.get("compound_name")


def test_load_collection_from_mgf_accepts_stream_and_path_objects():
    path = TESTDATA / "pesticides.mgf"
    from_path = load_collection_from_mgf(path)
    from_str = load_collection_from_mgf(str(path))
    with open(path, encoding="utf-8") as handle:
        from_stream = load_collection_from_mgf(StringIO(handle.read()))

    _assert_same_collection(from_str, from_path)
    _assert_same_collection(from_stream, from_path)


def test_load_collection_from_mgf_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_collection_from_mgf(str(tmp_path / "missing.mgf"))
    with pytest.raises(FileNotFoundError):
        load_collection_from_mgf(tmp_path / "missing.mgf")


def test_load_collection_from_mgf_without_spectra_raises(tmp_path):
    empty = tmp_path / "empty.mgf"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one Spectrum"):
        load_collection_from_mgf(empty)


def test_load_collection_from_mgf_binning_parameters():
    path = TESTDATA / "pesticides.mgf"
    collection = load_collection_from_mgf(path, mz_precision=0.01, mz_rounding="floor")
    expected = SpectraCollection(load_from_mgf(path), mz_precision=0.01)

    assert collection.mz_precision == 0.01
    assert collection.fragments.mz_precision == 0.01
    assert collection.fragments.mz_rounding == "floor"
    assert collection.n_bins <= expected.n_bins


def test_load_ms2_dataset_uses_collection_importer_for_mgf():
    path = TESTDATA / "pesticides.mgf"
    via_dataset = load_ms2_dataset(str(path), mz_precision=0.001)
    direct = load_collection_from_mgf(path, mz_precision=0.001)

    _assert_same_collection(via_dataset, direct)


def test_load_collection_from_mgf_keeps_pickydict_key_order_and_duplicates(tmp_path):
    content = (
        "BEGIN IONS\n"
        "PEPMASS=200.5 1000\n"
        "Precursor_MZ=200.5\n"
        "Compound Name=SuperStuff\n"
        "RTINSECONDS=12.5\n"
        "CHARGE=2+\n"
        "100.0 10.0\n"
        "150.0 20.0\n"
        "END IONS\n"
    )
    path = tmp_path / "keys.mgf"
    path.write_text(content, encoding="utf-8")

    for metadata_harmonization in (True, False):
        expected = SpectraCollection(load_from_mgf(path, metadata_harmonization=metadata_harmonization))
        collection = load_collection_from_mgf(path, metadata_harmonization=metadata_harmonization)
        _assert_same_collection(collection, expected)

    harmonized = load_collection_from_mgf(path)
    row = harmonized[0]
    assert row.get("compound_name") == "SuperStuff"
    assert row.get("retention_time") == 12.5
    assert row.get("charge") == 2
    assert row.get("precursor_mz") == 200.5


def test_harmonize_metadata_values_matches_metadata_class():
    raw = {
        "pepmass": (200.5, 1000.0),
        "charge": "2+",
        "ionmode": "Positive",
        "retention_time": "12.5",
        "parent_mass": "450.1",
        "compound_name": "n/a",
        "smiles": "CCO",
    }
    harmonized = harmonize_metadata_values(raw)

    metadata = Metadata(dict(raw))
    metadata.harmonize_values()
    assert harmonized == metadata.data

    assert harmonized["precursor_mz"] == 200.5
    assert harmonized["precursor_intensity"] == 1000.0
    assert harmonized["charge"] == 2
    assert harmonized["ionmode"] == "positive"
    assert harmonized["retention_time"] == 12.5
    assert harmonized["parent_mass"] == 450.1
    assert "compound_name" not in harmonized
    assert raw["charge"] == "2+"  # input is not modified


def test_from_peak_arrays_matches_spectrum_based_construction():
    spectra = [
        SpectrumBuilder()
        .with_mz(np.array([100.0, 150.0, 150.0000001]))
        .with_intensities(np.array([1.0, 2.0, 3.0]))
        .build(),
        SpectrumBuilder().with_mz(np.array([])).with_intensities(np.array([])).build(),
        SpectrumBuilder().with_mz(np.array([120.5])).with_intensities(np.array([5.0])).build(),
    ]
    expected = CSRFragmentCollection(spectra, mz_precision=0.01)

    lengths = np.array([3, 0, 1])
    mz = np.concatenate([s.mz for s in spectra])
    intensities = np.concatenate([s.intensities for s in spectra])
    built = CSRFragmentCollection.from_peak_arrays(lengths, mz, intensities, mz_precision=0.01)

    assert built.shape == expected.shape
    np.testing.assert_array_equal(built.array.indptr, expected.array.indptr)
    np.testing.assert_array_equal(built.array.indices, expected.array.indices)
    np.testing.assert_allclose(built.array.data, expected.array.data)


def test_from_peak_arrays_keeps_intensity_dtype():
    lengths = np.array([2, 1])
    mz = np.array([100.0, 200.0, 300.0])

    built = CSRFragmentCollection.from_peak_arrays(lengths, mz, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert built.array.data.dtype == np.float32

    built = CSRFragmentCollection.from_peak_arrays(lengths, mz, np.array([1, 2, 3]))
    assert built.array.data.dtype == np.float64
    np.testing.assert_array_equal(built.array.data, [1.0, 2.0, 3.0])


def test_from_peak_arrays_without_peaks_and_with_bad_input():
    empty = CSRFragmentCollection.from_peak_arrays(np.array([0, 0]), np.array([]), np.array([]))
    assert empty.shape == (2, 0)

    with pytest.raises(ValueError, match="lengths sum"):
        CSRFragmentCollection.from_peak_arrays(np.array([2]), np.array([100.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="same length"):
        CSRFragmentCollection.from_peak_arrays(np.array([1]), np.array([100.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="non-negative"):
        CSRFragmentCollection.from_peak_arrays(np.array([1]), np.array([-1.0]), np.array([1.0]))
