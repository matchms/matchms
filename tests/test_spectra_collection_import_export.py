import os
import numpy as np
import pytest
from matchms import SpectraCollection, Spectrum
from matchms.importing.load_spectra import load_ms2_dataset


def _testdata_file(filename):
    tests_root = os.path.join(os.path.dirname(__file__), "testdata")
    return os.path.join(tests_root, filename)


def _assert_collection_contains_spectra(collection, expected_num_spectra):
    assert isinstance(collection, SpectraCollection)
    assert len(collection) == expected_num_spectra

    for spectrum in collection:
        assert isinstance(spectrum, Spectrum)


@pytest.mark.parametrize(
    "filename, expected_num_spectra",
    [
        ("pesticides.mgf", 76),
        ("testdata.mgf", 30),
        ("testdata.mzml", 10),
        ("testdata.mzXML", 1),
        ("massbank_five_spectra.msp", 5),
    ],
)
def test_load_ms2_dataset_returns_spectra_collection(filename, expected_num_spectra):
    file = _testdata_file(filename)

    collection = load_ms2_dataset(file)

    _assert_collection_contains_spectra(collection, expected_num_spectra)


@pytest.mark.parametrize(
    "filename, ftype, expected_num_spectra",
    [
        ("testdata.mgf", "auto", 30),
        ("testdata.mgf", "mgf", 30),
        ("testdata.mzml", "auto", 10),
        ("testdata.mzml", "mzml", 10),
        ("testdata.mzXML", "auto", 1),
        ("testdata.mzXML", "mzxml", 1),
        ("massbank_five_spectra.msp", "auto", 5),
        ("massbank_five_spectra.msp", "msp", 5),
    ],
)
def test_load_ms2_dataset_with_auto_and_explicit_filetype(filename, ftype, expected_num_spectra):
    file = _testdata_file(filename)

    collection = load_ms2_dataset(file, ftype=ftype)

    _assert_collection_contains_spectra(collection, expected_num_spectra)


def test_spectra_collection_to_mgf_roundtrip(tmp_path):
    input_file = _testdata_file("massbank_five_spectra.msp")
    output_file = tmp_path / "exported.mgf"

    collection = load_ms2_dataset(input_file)
    collection.to_mgf(str(output_file))

    assert output_file.exists()

    imported = load_ms2_dataset(str(output_file))

    assert isinstance(imported, SpectraCollection)
    assert len(imported) == len(collection)


def test_spectra_collection_to_msp_roundtrip(tmp_path):
    input_file = _testdata_file("testdata.mgf")
    output_file = tmp_path / "exported.msp"

    collection = load_ms2_dataset(input_file)
    collection.to_msp(str(output_file))

    assert output_file.exists()

    imported = load_ms2_dataset(str(output_file))

    assert isinstance(imported, SpectraCollection)
    assert len(imported) == len(collection)


def test_spectra_collection_to_json_roundtrip(tmp_path):
    input_file = _testdata_file("testdata.mgf")
    output_file = tmp_path / "exported.json"

    collection = load_ms2_dataset(input_file)
    collection.to_json(str(output_file))

    assert output_file.exists()

    imported = load_ms2_dataset(str(output_file))

    assert isinstance(imported, SpectraCollection)
    assert len(imported) == len(collection)


@pytest.mark.parametrize(
    "export_method, extension",
    [
        ("to_mgf", "mgf"),
        ("to_msp", "msp"),
        ("to_json", "json"),
    ],
)
def test_spectra_collection_export_does_not_overwrite_existing_file(
    tmp_path,
    export_method,
    extension,
):
    input_file = _testdata_file("massbank_five_spectra.msp")
    output_file = tmp_path / f"exported.{extension}"
    output_file.write_text("already exists", encoding="utf-8")

    collection = load_ms2_dataset(input_file)

    with pytest.raises(FileExistsError):
        getattr(collection, export_method)(str(output_file))


@pytest.mark.parametrize(
    "export_method, extension",
    [
        ("to_mgf", "mgf"),
        ("to_msp", "msp"),
    ],
)
def test_spectra_collection_export_append_supported_for_mgf_and_msp(
    tmp_path,
    export_method,
    extension,
):
    input_file = _testdata_file("massbank_five_spectra.msp")
    output_file = tmp_path / f"exported.{extension}"

    collection = load_ms2_dataset(input_file)

    getattr(collection, export_method)(str(output_file))
    getattr(collection, export_method)(str(output_file), append=True)

    imported = load_ms2_dataset(str(output_file))

    assert isinstance(imported, SpectraCollection)
    assert len(imported) == 2 * len(collection)


def test_spectra_collection_to_json_append_not_supported(tmp_path):
    input_file = _testdata_file("massbank_five_spectra.msp")
    output_file = tmp_path / "exported.json"

    collection = load_ms2_dataset(input_file)
    collection.to_json(str(output_file))

    with pytest.raises(ValueError, match="Appending is not supported|append"):
        collection.to_json(str(output_file), append=True)


def test_load_ms2_dataset_passes_mz_precision_to_collection():
    file = _testdata_file("testdata.mgf")

    collection = load_ms2_dataset(
        file,
        mz_precision=0.01,
    )

    assert isinstance(collection, SpectraCollection)
    assert collection.fragments.mz_precision == 0.01
    assert collection.fragments.mz_rounding == "round"

    spectrum = collection[0]

    # All reconstructed m/z values should lie on the 0.01 grid.
    scaled_mz = spectrum.peaks.mz / 0.01
    assert np.allclose(scaled_mz, np.round(scaled_mz))
    assert spectrum.peaks.mz[0] == pytest.approx(66.14)  # actual peak in the file is 66.137428
