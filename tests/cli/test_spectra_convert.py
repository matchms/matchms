import json
import os
import pytest
from matchms.cli.main import build_parser, main
from matchms.exporting.save_spectra import SUPPORTED_FILE_FORMATS as OUTPUT_FORMATS
from matchms.importing import load_ms2_dataset
from matchms.importing.load_spectra import SUPPORTED_FILE_FORMATS as INPUT_FORMATS


TEST_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testdata"))
MGF_FILE = os.path.join(TEST_DATA, "testdata.mgf")
MSP_FILE = os.path.join(TEST_DATA, "Hydrogen_chloride.msp")


def run_cli(*argv, capsys=None):
    """Run the CLI and return (exit_code, parsed_json_stdout)."""
    exit_code = main(list(argv))
    out = capsys.readouterr().out
    return exit_code, json.loads(out)


def load_n(file, ftype=None):
    kwargs = {} if ftype is None else {"ftype": ftype}
    return len(load_ms2_dataset(file, **kwargs))


def test_spectra_convert_mgf_to_msp(tmp_path, capsys):
    out = tmp_path / "out.msp"

    exit_code, payload = run_cli("spectra", "convert", MGF_FILE, str(out), "--json", capsys=capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["operation"] == "spectra convert"
    assert payload["input_file"] == MGF_FILE
    assert payload["input_format"] == "mgf"
    assert payload["output_file"] == str(out)
    assert payload["output_format"] == "msp"
    assert payload["n_spectra"] == 30
    # output file must be a valid MSP with the same number of spectra
    assert out.exists()
    assert load_n(str(out), ftype="msp") == 30


@pytest.mark.parametrize(
    "src, dst_ext",
    [
        ("mgf", "msp"),
        ("mgf", "json"),
        ("mgf", "pickle"),
        ("msp", "mgf"),
        ("msp", "json"),
    ],
)
def test_spectra_convert_matrix(tmp_path, capsys, src, dst_ext):
    src_file = MGF_FILE if src == "mgf" else MSP_FILE
    n_src = 30 if src == "mgf" else 1
    out = tmp_path / f"out.{dst_ext}"

    exit_code, payload = run_cli("spectra", "convert", src_file, str(out), "--json", capsys=capsys)

    assert exit_code == 0
    assert payload["output_format"] == dst_ext
    assert payload["n_spectra"] == n_src
    assert out.exists()
    assert load_n(str(out), ftype=dst_ext) == n_src


def test_spectra_convert_explicit_ftype(tmp_path, capsys):
    out = tmp_path / "out.msp"

    exit_code, payload = run_cli(
        "spectra",
        "convert",
        MGF_FILE,
        str(out),
        "--ftype",
        "mgf",
        "--json",
        capsys=capsys,
    )

    assert exit_code == 0
    assert payload["n_spectra"] == 30


def test_spectra_convert_table_output(tmp_path, capsys):
    out = tmp_path / "out.msp"

    exit_code = main(["spectra", "convert", MGF_FILE, str(out), "--table"])
    text = capsys.readouterr().out

    assert exit_code == 0
    assert "Converted" in text
    assert str(out) in text
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


def test_spectra_convert_missing_input(tmp_path, capsys):
    missing = tmp_path / "nope.mgf"
    out = tmp_path / "out.msp"

    exit_code, payload = run_cli("spectra", "convert", str(missing), str(out), "--json", capsys=capsys)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "file_not_found"
    assert payload["input_file"] == str(missing)
    assert set(payload["valid_values"]) == set(INPUT_FORMATS)
    assert not out.exists()


@pytest.mark.parametrize(
    "bad_ext",
    ["txt", "csv", "tsv", "fasta"],
    ids=lambda v: f"unsupported_input_{v}",
)
def test_spectra_convert_unsupported_input_extension(tmp_path, capsys, bad_ext):
    bad = tmp_path / f"in.{bad_ext}"
    bad.write_text("junk\n", encoding="utf-8")
    out = tmp_path / "out.msp"

    exit_code, payload = run_cli("spectra", "convert", str(bad), str(out), "--json", capsys=capsys)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "unsupported_input_format"
    assert set(payload["valid_values"]) == set(INPUT_FORMATS)
    assert not out.exists()


def test_spectra_convert_unsupported_output_extension(tmp_path, capsys):
    out = tmp_path / "out.txt"

    exit_code, payload = run_cli("spectra", "convert", MGF_FILE, str(out), "--json", capsys=capsys)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "unsupported_output_format"
    assert set(payload["valid_values"]) == set(OUTPUT_FORMATS)
    assert not out.exists()


def test_spectra_convert_output_already_exists(tmp_path, capsys):
    out = tmp_path / "out.msp"
    out.write_text("existing\n", encoding="utf-8")

    exit_code, payload = run_cli("spectra", "convert", MGF_FILE, str(out), "--json", capsys=capsys)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "file_exists"
    # output file must be left untouched
    assert out.read_text(encoding="utf-8") == "existing\n"


def test_spectra_convert_append(tmp_path, capsys):
    out = tmp_path / "out.msp"

    first, _ = run_cli("spectra", "convert", MGF_FILE, str(out), "--json", capsys=capsys)
    second, payload = run_cli("spectra", "convert", MGF_FILE, str(out), "--append", "--json", capsys=capsys)

    assert first == 0
    assert second == 0
    assert payload["n_spectra"] == 30
    # appended, so the file now holds 30 + 30 spectra
    assert load_n(str(out), ftype="msp") == 60


def test_spectra_convert_append_not_supported_for_json(tmp_path, capsys):
    out = tmp_path / "out.json"

    exit_code, payload = run_cli("spectra", "convert", MGF_FILE, str(out), "--append", "--json", capsys=capsys)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "append"
    assert set(payload["valid_values"]) == {"mgf", "msp"}
    assert not out.exists()


def test_spectra_convert_export_style_gnps(tmp_path, capsys):
    out = tmp_path / "out.json"

    exit_code, payload = run_cli(
        "spectra",
        "convert",
        MSP_FILE,
        str(out),
        "--export-style",
        "gnps",
        "--json",
        capsys=capsys,
    )

    assert exit_code == 0
    assert payload["n_spectra"] == 1
    assert out.exists()


def test_spectra_convert_pickle_requires_matchms_style(tmp_path, capsys):
    out = tmp_path / "out.pickle"

    exit_code, payload = run_cli(
        "spectra",
        "convert",
        MGF_FILE,
        str(out),
        "--export-style",
        "gnps",
        "--json",
        capsys=capsys,
    )

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "export_style"
    assert not out.exists()


def test_spectra_convert_invalid_export_style(tmp_path, capsys):
    out = tmp_path / "out.json"

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["spectra", "convert", MGF_FILE, str(out), "--export-style", "bogus"])
