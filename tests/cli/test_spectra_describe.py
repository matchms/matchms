import json
import os
import pytest
from matchms.cli.main import build_parser, main
from matchms.exporting import save_as_mgf
from matchms.importing.load_spectra import SUPPORTED_FILE_FORMATS as INPUT_FORMATS


TEST_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testdata"))
MGF_FILE = os.path.join(TEST_DATA, "testdata.mgf")
MSP_FILE = os.path.join(TEST_DATA, "Hydrogen_chloride.msp")

STATS_ROWS = ("count", "mean", "std", "min", "25%", "50%", "75%", "max")
STATS_METRICS = ("peak_counts", "intensity_sums", "intensity_entropy")


def run_cli(*argv, capsys=None):
    """Run the CLI and return (exit_code, parsed_json_stdout)."""
    exit_code = main(list(argv))
    out = capsys.readouterr().out
    payload = json.loads(out)
    return exit_code, payload


def test_spectra_describe_json(tmp_path, capsys):
    exit_code, payload = run_cli("spectra", "describe", MGF_FILE, "--json", capsys=capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["operation"] == "spectra describe"
    assert payload["input_file"] == MGF_FILE
    assert payload["n_spectra"] == 30
    assert set(payload["statistics"]) == set(STATS_METRICS)
    for metric in STATS_METRICS:
        assert set(payload["statistics"][metric]) == set(STATS_ROWS)
        values = payload["statistics"][metric]
        assert values["count"] == 30
        assert all(isinstance(v, float) for v in values.values())
        assert values["min"] <= values["mean"] <= values["max"]


def test_spectra_describe_matches_describe_method(tmp_path, capsys):
    """The CLI must report exactly what SpectraCollection.describe() computes."""
    from matchms.importing import load_ms2_dataset

    stats = load_ms2_dataset(MGF_FILE).describe()
    exit_code, payload = run_cli("spectra", "describe", MGF_FILE, "--json", capsys=capsys)

    assert exit_code == 0
    for metric in STATS_METRICS:
        for row in STATS_ROWS:
            expected = float(stats.loc[row, metric])
            assert payload["statistics"][metric][row] == pytest.approx(expected)


@pytest.mark.parametrize("spectrumfile", [MGF_FILE, MSP_FILE])
def test_spectra_describe_input_formats(tmp_path, capsys, spectrumfile):
    expected_n = 30 if spectrumfile.endswith(".mgf") else 1
    exit_code, payload = run_cli("spectra", "describe", spectrumfile, "--json", capsys=capsys)

    assert exit_code == 0
    assert payload["n_spectra"] == expected_n
    assert payload["statistics"]["peak_counts"]["count"] == expected_n


def test_spectra_describe_table_output(tmp_path, capsys):
    exit_code = main(["spectra", "describe", MSP_FILE, "--table"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "SpectraCollection Describe" in out
    assert "peak_counts" in out
    assert "intensity_entropy" in out
    assert "50%" in out
    # forced --table must not emit JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_spectra_describe_missing_file(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.mgf"

    exit_code, payload = run_cli("spectra", "describe", str(missing), "--json", capsys=capsys)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "file_not_found"
    assert payload["input_file"] == str(missing)
    assert set(payload["valid_values"]) == set(INPUT_FORMATS)


def test_spectra_describe_unsupported_extension(tmp_path, capsys):
    fake = tmp_path / "spectra.txt"
    fake.write_text("not a spectra file\n", encoding="utf-8")

    exit_code, payload = run_cli("spectra", "describe", str(fake), "--json", capsys=capsys)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "unsupported_input_format"
    assert set(payload["valid_values"]) == set(INPUT_FORMATS)


def test_spectra_describe_requires_spectrumfile(tmp_path, capsys):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["spectra", "describe"])

    with pytest.raises(SystemExit):
        parser.parse_args(["spectra", "describe", MGF_FILE, MGF_FILE])


def test_spectra_describe_empty_collection(tmp_path, capsys):
    empty = tmp_path / "empty.mgf"
    save_as_mgf([], str(empty), "matchms", file_mode="w")

    exit_code, payload = run_cli("spectra", "describe", str(empty), "--json", capsys=capsys)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "empty_spectra"
    assert payload["input_file"] == str(empty)
