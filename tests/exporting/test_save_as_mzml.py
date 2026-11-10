import os
import numpy as np
import pytest
from pyteomics import mzml
from matchms import Spectrum
from matchms.exporting import save_as_mzml, save_spectra
from matchms.exporting.save_spectra import SUPPORTED_FILE_FORMATS
from matchms.importing import load_from_mzml
from ..builder_spectrum import SpectrumBuilder


def _spectrum(metadata: dict, mz=None, intensities=None) -> Spectrum:
    builder = (
        SpectrumBuilder()
        .with_mz(np.array([100.123456789, 200.0, 300.5]) if mz is None else mz)
        .with_intensities(np.array([0.123456789, 10.0, 500.0]) if intensities is None else intensities)
    )
    return builder.with_metadata(metadata, metadata_harmonization=False).build()


def _raw_spectra(filename) -> list[dict]:
    """Read the file with pyteomics, as written (no matchms metadata parsing)."""
    with mzml.read(str(filename)) as reader:
        return list(reader)


def _first(value):
    """The current loader returns the scan start time as a list; a float is accepted as well."""
    return float(np.asarray(value).ravel()[0])


def test_save_as_mzml_round_trip_peaks_and_parsed_metadata(tmp_path):
    spectrum = _spectrum(
        {"charge": -1, "precursor_mz": 301.2, "ionmode": "negative", "retention_time": 61.5, "title": "my title"}
    )
    filename = tmp_path / "test.mzML"

    save_as_mzml(spectrum, filename)
    loaded = list(load_from_mzml(filename))

    assert len(loaded) == 1
    np.testing.assert_array_equal(loaded[0].peaks.mz, spectrum.peaks.mz)
    np.testing.assert_array_equal(loaded[0].peaks.intensities, spectrum.peaks.intensities)
    assert loaded[0].get("charge") == -1
    assert loaded[0].get("precursor_mz") == 301.2
    assert loaded[0].get("title") == "my title"
    assert _first(loaded[0].get("scan_start_time")) == 61.5


def test_save_as_mzml_writes_controlled_vocabulary_terms_and_user_params(tmp_path):
    spectrum = _spectrum(
        {
            "charge": 2,
            "precursor_mz": 301.2,
            "precursor_intensity": 12345.0,
            "ionmode": "positive",
            "retention_time": 61.5,
            "collision_energy": "35",
            "ms_level": "MS2",
            "compound_name": "test compound",
            "smiles": "CCO",
            "adduct": "[M+2H]2+",
            "parent_mass": 600.4,
        }
    )
    filename = tmp_path / "terms.mzML"

    save_as_mzml(spectrum, filename)
    raw = _raw_spectra(filename)[0]

    selected_ion = raw["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]
    assert selected_ion["selected ion m/z"] == 301.2
    assert selected_ion["peak intensity"] == 12345.0
    assert selected_ion["charge state"] == 2
    assert raw["precursorList"]["precursor"][0]["activation"]["collision energy"] == 35.0
    assert raw["ms level"] == 2
    assert "positive scan" in raw
    scan_start_time = raw["scanList"]["scan"][0]["scan start time"]
    assert scan_start_time == 61.5
    assert scan_start_time.unit_info == "second"
    # Remaining metadata is kept as userParams under the matchms key names.
    assert raw["compound_name"] == "test compound"
    assert raw["smiles"] == "CCO"
    assert raw["adduct"] == "[M+2H]2+"
    assert raw["parent_mass"] == 600.4
    # Entries written as terms are not duplicated as userParams.
    assert "precursor_mz" not in raw
    assert "ionmode" not in raw
    assert "retention_time" not in raw


def test_save_as_mzml_retention_time_in_minutes(tmp_path):
    spectrum = _spectrum({"retention_time": 1.5})
    filename = tmp_path / "minutes.mzML"

    save_as_mzml(spectrum, filename, retention_time_unit="minute")

    scan_start_time = _raw_spectra(filename)[0]["scanList"]["scan"][0]["scan start time"]
    assert scan_start_time == 1.5
    assert scan_start_time.unit_info == "minute"


def test_save_as_mzml_rejects_unknown_retention_time_unit_and_ms_level(tmp_path):
    spectrum = _spectrum({})

    with pytest.raises(ValueError, match="retention_time_unit"):
        save_as_mzml(spectrum, tmp_path / "unit.mzML", retention_time_unit="hour")
    with pytest.raises(ValueError, match="ms_level"):
        save_as_mzml(spectrum, tmp_path / "level.mzML", ms_level=0)


def test_save_as_mzml_ms_level_default_and_from_metadata(tmp_path):
    ms2_default = _spectrum({"precursor_mz": 301.2})
    ms1 = _spectrum({"ms_level": 1})
    ms3 = _spectrum({"ms_level": "3", "precursor_mz": 150.1})
    filename = tmp_path / "levels.mzML"

    save_as_mzml([ms2_default, None, ms1, ms3], filename)

    assert [raw["ms level"] for raw in _raw_spectra(filename)] == [2, 1, 3]
    assert len(list(load_from_mzml(filename, ms_level=1))) == 1
    assert len(list(load_from_mzml(filename, ms_level=2))) == 1
    assert len(list(load_from_mzml(filename, ms_level=3))) == 1


def test_save_as_mzml_polarity_from_charge_when_ionmode_missing(tmp_path):
    negative = _spectrum({"charge": -1, "precursor_mz": 301.2})
    unknown = _spectrum({"precursor_mz": 301.2})
    filename = tmp_path / "polarity.mzML"

    save_as_mzml([negative, unknown], filename)
    raw = _raw_spectra(filename)

    assert "negative scan" in raw[0]
    assert "negative scan" not in raw[1] and "positive scan" not in raw[1]
    assert [spectrum.get("charge") for spectrum in load_from_mzml(filename)] == [-1, None]


def test_save_as_mzml_keeps_non_numeric_collision_energy_as_user_param(tmp_path):
    spectrum = _spectrum({"precursor_mz": 301.2, "collision_energy": "ramp 20-40 eV"})
    filename = tmp_path / "ramp.mzML"

    save_as_mzml(spectrum, filename)
    raw = _raw_spectra(filename)[0]

    assert raw["collision_energy"] == "ramp 20-40 eV"
    assert "collision energy" not in raw["precursorList"]["precursor"][0]["activation"]


def test_save_as_mzml_export_style_renames_user_params(tmp_path):
    spectrum = _spectrum({"precursor_mz": 301.2, "compound_name": "test compound", "smiles": "CCO"})
    filename = tmp_path / "gnps.mzML"

    save_as_mzml(spectrum, filename, export_style="gnps")
    raw = _raw_spectra(filename)[0]

    assert raw["NAME"] == "test compound"
    assert raw["SMILES"] == "CCO"
    assert "compound_name" not in raw
    # The selected ion is written as a term, not as the GNPS key PEPMASS.
    assert "PEPMASS" not in raw
    assert raw["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]["selected ion m/z"] == 301.2


def test_save_as_mzml_skips_fingerprint_and_peak_comments(tmp_path):
    spectrum = _spectrum({"precursor_mz": 301.2, "fingerprint": np.array([0, 1, 1]), "peak_comments": {100.0: "a"}})
    filename = tmp_path / "skip.mzML"

    save_as_mzml(spectrum, filename)
    raw = _raw_spectra(filename)[0]

    assert "fingerprint" not in raw
    assert "peak_comments" not in raw


def test_save_as_mzml_existing_file_round_trip(tmp_path):
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra = list(load_from_mzml(os.path.join(module_root, "testdata", "testdata.mzml")))
    filename = tmp_path / "copy.mzML"

    save_as_mzml(spectra, filename)
    loaded = list(load_from_mzml(filename))

    assert len(loaded) == len(spectra) == 10
    for original, copy in zip(spectra, loaded, strict=True):
        np.testing.assert_array_equal(copy.peaks.mz, original.peaks.mz)
        np.testing.assert_array_equal(copy.peaks.intensities, original.peaks.intensities)
        assert copy.get("precursor_mz") == original.get("precursor_mz")
        assert copy.get("charge") == original.get("charge")
        assert copy.get("title") == original.get("title")


def test_save_spectra_dispatches_mzml_extension(tmp_path):
    spectrum = _spectrum({"precursor_mz": 301.2, "charge": 1})
    filename = tmp_path / "dispatch.mzml"

    save_spectra([spectrum], str(filename))

    assert "mzml" in SUPPORTED_FILE_FORMATS
    loaded = list(load_from_mzml(filename))
    assert len(loaded) == 1
    assert loaded[0].get("precursor_mz") == 301.2
