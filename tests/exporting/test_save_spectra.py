import importlib
import pickle
import numpy as np
import pytest
from matchms import Spectrum
from matchms.exporting.save_spectra import save_as_pickled_file, save_spectra


save_spectra_module = importlib.import_module("matchms.exporting.save_spectra")


def _make_spectrum():
    return Spectrum(
        mz=np.array([100.0, 200.0], dtype="float"),
        intensities=np.array([0.5, 1.0], dtype="float"),
        metadata={"precursor_mz": 201.0, "compound_name": "test compound"},
    )


def test_save_spectra_unknown_file_extension(tmp_path):
    spectrum = _make_spectrum()
    filename = tmp_path / "spectra.unknown"

    with pytest.raises(TypeError, match="File extension"):
        save_spectra([spectrum], str(filename))


def test_save_spectra_does_not_overwrite_existing_file(tmp_path):
    spectrum = _make_spectrum()
    filename = tmp_path / "spectra.mgf"
    filename.write_text("already exists", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_spectra([spectrum], str(filename))


@pytest.mark.parametrize("extension", ["json", "pickle"])
def test_save_spectra_append_raises_for_unsupported_filetypes(tmp_path, extension):
    spectrum = _make_spectrum()
    filename = tmp_path / f"spectra.{extension}"

    with pytest.raises(ValueError, match="append"):
        save_spectra([spectrum], str(filename), append=True)


def test_save_spectra_empty_list_creates_empty_file(tmp_path):
    filename = tmp_path / "empty.mgf"

    save_spectra([], str(filename))

    assert filename.exists()
    assert filename.read_text(encoding="utf-8") == ""


def test_save_spectra_json_dispatches_to_save_as_json(monkeypatch, tmp_path):
    spectrum = _make_spectrum()
    filename = tmp_path / "spectra.json"
    calls = {}

    def mock_save_as_json(spectra, file, export_style):
        calls["spectra"] = spectra
        calls["file"] = file
        calls["export_style"] = export_style

    monkeypatch.setattr(save_spectra_module, "save_as_json", mock_save_as_json)

    save_spectra([spectrum], str(filename), export_style="gnps")

    assert calls["spectra"] == [spectrum]
    assert calls["file"] == str(filename)
    assert calls["export_style"] == "gnps"


@pytest.mark.parametrize(
    "append, expected_file_mode",
    [
        (False, "w"),
        (True, "a"),
    ],
)
def test_save_spectra_mgf_dispatches_with_file_mode(
    monkeypatch,
    tmp_path,
    append,
    expected_file_mode,
):
    spectrum = _make_spectrum()
    filename = tmp_path / "spectra.mgf"
    calls = {}

    if append:
        filename.write_text("existing content\n", encoding="utf-8")

    def mock_save_as_mgf(spectra, file, export_style, file_mode="w"):
        calls["spectra"] = spectra
        calls["file"] = file
        calls["export_style"] = export_style
        calls["file_mode"] = file_mode

    monkeypatch.setattr(save_spectra_module, "save_as_mgf", mock_save_as_mgf)

    save_spectra([spectrum], str(filename), export_style="matchms", append=append)

    assert calls["spectra"] == [spectrum]
    assert calls["file"] == str(filename)
    assert calls["export_style"] == "matchms"
    assert calls["file_mode"] == expected_file_mode


@pytest.mark.parametrize(
    "append, expected_file_mode",
    [
        (False, "w"),
        (True, "a"),
    ],
)
def test_save_spectra_msp_dispatches_with_file_mode(
    monkeypatch,
    tmp_path,
    append,
    expected_file_mode,
):
    spectrum = _make_spectrum()
    filename = tmp_path / "spectra.msp"
    calls = {}

    if append:
        filename.write_text("existing content\n", encoding="utf-8")

    def mock_save_as_msp(spectra, file, style="matchms", file_mode="w"):
        calls["spectra"] = spectra
        calls["file"] = file
        calls["style"] = style
        calls["file_mode"] = file_mode

    monkeypatch.setattr(save_spectra_module, "save_as_msp", mock_save_as_msp)

    save_spectra([spectrum], str(filename), export_style="nist", append=append)

    assert calls["spectra"] == [spectrum]
    assert calls["file"] == str(filename)
    assert calls["style"] == "nist"
    assert calls["file_mode"] == expected_file_mode


def test_save_spectra_accepts_single_spectrum(monkeypatch, tmp_path):
    spectrum = _make_spectrum()
    filename = tmp_path / "single.mgf"
    calls = {}

    def mock_save_as_mgf(spectra, file, export_style, file_mode="w"):
        calls["spectra"] = spectra
        calls["file"] = file
        calls["export_style"] = export_style
        calls["file_mode"] = file_mode

    monkeypatch.setattr(save_spectra_module, "save_as_mgf", mock_save_as_mgf)

    save_spectra(spectrum, str(filename))

    assert calls["spectra"] == [spectrum]
    assert calls["file"] == str(filename)
    assert calls["export_style"] == "matchms"
    assert calls["file_mode"] == "w"


def test_save_spectra_pickle_export(tmp_path):
    spectrum = _make_spectrum()
    filename = tmp_path / "spectra.pickle"

    save_spectra([spectrum], str(filename))

    assert filename.exists()

    with open(filename, "rb") as file:
        loaded = pickle.load(file)

    assert len(loaded) == 1
    assert isinstance(loaded[0], Spectrum)
    assert loaded[0] == spectrum


def test_save_as_pickled_file_requires_list(tmp_path):
    spectrum = _make_spectrum()
    filename = tmp_path / "spectra.pickle"

    with pytest.raises(TypeError, match="Expected list of spectra"):
        save_as_pickled_file(spectrum, str(filename))


def test_save_as_pickled_file_requires_list_of_spectra(tmp_path):
    filename = tmp_path / "spectra.pickle"

    with pytest.raises(TypeError, match="Expected list of spectra"):
        save_as_pickled_file(["not a spectrum"], str(filename))


def test_save_as_pickled_file_does_not_overwrite_existing_file(tmp_path):
    spectrum = _make_spectrum()
    filename = tmp_path / "spectra.pickle"
    filename.write_bytes(b"already exists")

    with pytest.raises(FileExistsError):
        save_as_pickled_file([spectrum], str(filename))