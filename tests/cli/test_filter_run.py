import json
import os
import pytest
from matchms.cli.main import build_parser, main
from matchms.exporting.save_spectra import SUPPORTED_FILE_FORMATS as OUTPUT_FORMATS
from matchms.filtering.filter_order import ALL_FILTERS
from matchms.importing import load_ms2_dataset
from matchms.importing.load_spectra import SUPPORTED_FILE_FORMATS as INPUT_FORMATS


TEST_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testdata"))
MGF_FILE = os.path.join(TEST_DATA, "testdata.mgf")
MSP_FILE = os.path.join(TEST_DATA, "Hydrogen_chloride.msp")

N_MGF = 30
N_MSP = 1


def run_cli(*argv, capsys=None):
    """Run the CLI and return (exit_code, parsed_json_stdout)."""
    exit_code = main(list(argv))
    out = capsys.readouterr().out
    payload = json.loads(out)
    return exit_code, payload


def load_n(file, ftype=None):
    kwargs = {} if ftype is None else {"ftype": ftype}
    if not os.path.exists(file) or os.path.getsize(file) == 0:
        return 0
    return len(load_ms2_dataset(file, **kwargs))


def filter_names(payload):
    return [f["name"] for f in payload["filters"]]


def assert_canonical_order(payload):
    """The reported filters must be sorted by the canonical matchms order."""
    order = {name: i for i, name in enumerate(f.__name__ for f in ALL_FILTERS)}
    indices = [order[name] for name in filter_names(payload)]
    assert indices == sorted(indices)


# -- basic run --------------------------------------------------------------


def test_filter_run_default_filters_json(tmp_path, capsys):
    out = tmp_path / "out.msp"

    exit_code, payload = run_cli(
        "filter", "run", MGF_FILE, str(out), "--json", capsys=capsys
    )

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["operation"] == "filter run"
    assert payload["input_file"] == MGF_FILE
    assert payload["input_format"] == "mgf"
    assert payload["output_file"] == str(out)
    assert payload["output_format"] == "msp"
    # no --pipeline -> DEFAULT_FILTERS
    assert payload["pipeline"] == "DEFAULT_FILTERS"
    assert payload["n_spectra_in"] == N_MGF
    assert payload["n_spectra_out"] >= 1
    assert payload["n_removed"] == payload["n_spectra_in"] - payload["n_spectra_out"]
    # no --report -> no report artifact
    assert "report_file" not in payload
    assert "report" not in payload
    assert_canonical_order(payload)
    # output file is a valid MSP and carries exactly the reported spectra
    assert out.exists()
    assert load_n(str(out), ftype="msp") == payload["n_spectra_out"]


def test_filter_run_default_pipeline_is_default_filters(tmp_path, capsys):
    from matchms.filtering.default_pipelines import DEFAULT_FILTERS

    out = tmp_path / "out.msp"
    exit_code, payload = run_cli(
        "filter", "run", MGF_FILE, str(out), "--json", capsys=capsys
    )
    assert exit_code == 0

    expected = {
        entry[0].__name__ if isinstance(entry, (tuple, list)) else entry.__name__
        for entry in DEFAULT_FILTERS
    }
    assert set(filter_names(payload)) == expected


def test_filter_run_table_output(tmp_path, capsys):
    out = tmp_path / "out.msp"

    exit_code = main(["filter", "run", MGF_FILE, str(out), "--table"])
    text = capsys.readouterr().out

    assert exit_code == 0
    assert "Filter run:" in text
    assert "pipeline" in text
    assert "DEFAULT_FILTERS" in text
    assert "Filters (in execution order):" in text
    assert "make_charge_int" in text
    # forced --table must not emit JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


def test_filter_run_msp_input(tmp_path, capsys):
    out = tmp_path / "out.mgf"

    exit_code, payload = run_cli(
        "filter", "run", MSP_FILE, str(out), "--json", capsys=capsys
    )

    assert exit_code == 0
    assert payload["n_spectra_in"] == N_MSP
    assert out.exists()
    assert load_n(str(out), ftype="mgf") == payload["n_spectra_out"]


# -- --pipeline -------------------------------------------------------------


def test_filter_run_pipeline_basic(tmp_path, capsys):
    from matchms.filtering.default_pipelines import BASIC_FILTERS

    out = tmp_path / "out.msp"
    exit_code, payload = run_cli(
        "filter", "run", MGF_FILE, str(out), "--pipeline", "BASIC_FILTERS", "--json",
        capsys=capsys,
    )

    assert exit_code == 0
    assert payload["pipeline"] == "BASIC_FILTERS"
    expected = {
        entry[0].__name__ if isinstance(entry, (tuple, list)) else entry.__name__
        for entry in BASIC_FILTERS
    }
    assert set(filter_names(payload)) == expected


def test_filter_run_invalid_pipeline(tmp_path, capsys):
    out = tmp_path / "out.msp"
    # --pipeline is validated by argparse choices -> SystemExit(2)
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["filter", "run", MGF_FILE, str(out), "--pipeline", "NOPE"]
        )


# -- --filter and --param ---------------------------------------------------


def test_filter_run_add_filter_and_params(tmp_path, capsys):
    out = tmp_path / "out.msp"
    exit_code, payload = run_cli(
        "filter",
        "run",
        MGF_FILE,
        str(out),
        "--pipeline",
        "BASIC_FILTERS",
        "--filter",
        "select_by_mz",
        "--param",
        "select_by_mz.mz_from=50",
        "--param",
        "select_by_mz.mz_to=300",
        "--json",
        capsys=capsys,
    )

    assert exit_code == 0
    names = filter_names(payload)
    # the added filter is present and ordered canonically
    assert "select_by_mz" in names
    assert_canonical_order(payload)
    # parameters are bound to the added filter with typed values
    by_name = {f["name"]: f for f in payload["filters"]}
    assert by_name["select_by_mz"]["parameters"] == {"mz_from": 50, "mz_to": 300}
    assert isinstance(by_name["select_by_mz"]["parameters"]["mz_from"], int)


def test_filter_run_param_value_typing(tmp_path, capsys):
    out = tmp_path / "out.msp"
    exit_code, payload = run_cli(
        "filter",
        "run",
        MGF_FILE,
        str(out),
        "--filter",
        "select_by_mz",
        "--param",
        "select_by_mz.mz_from=50.5",
        "--json",
        capsys=capsys,
    )

    assert exit_code == 0
    by_name = {f["name"]: f for f in payload["filters"]}
    # 50.5 is a float, not an int or string
    assert by_name["select_by_mz"]["parameters"]["mz_from"] == 50.5
    assert isinstance(by_name["select_by_mz"]["parameters"]["mz_from"], float)


def test_filter_run_param_json_list_value(tmp_path, capsys):
    out = tmp_path / "out.msp"
    exit_code, payload = run_cli(
        "filter",
        "run",
        MGF_FILE,
        str(out),
        "--filter",
        "harmonize_missing_entries",
        "--param",
        'harmonize_missing_entries.keys=["inchi", "smiles"]',
        "--json",
        capsys=capsys,
    )

    assert exit_code == 0
    by_name = {f["name"]: f for f in payload["filters"]}
    assert by_name["harmonize_missing_entries"]["parameters"]["keys"] == [
        "inchi",
        "smiles",
    ]


def test_filter_run_removes_all_spectra(tmp_path, capsys):
    out = tmp_path / "out.msp"
    exit_code, payload = run_cli(
        "filter",
        "run",
        MGF_FILE,
        str(out),
        "--filter",
        "require_minimum_number_of_peaks",
        "--param",
        "require_minimum_number_of_peaks.n_required=100000",
        "--json",
        capsys=capsys,
    )

    assert exit_code == 0
    assert payload["n_spectra_in"] == N_MGF
    assert payload["n_spectra_out"] == 0
    assert payload["n_removed"] == N_MGF
    # the output file exists but is empty
    assert out.exists()
    assert load_n(str(out), ftype="msp") == 0


# -- --report ---------------------------------------------------------------


def test_filter_run_report(tmp_path, capsys):
    out = tmp_path / "out.msp"
    exit_code, payload = run_cli(
        "filter",
        "run",
        MGF_FILE,
        str(out),
        "--filter",
        "require_minimum_number_of_peaks",
        "--param",
        "require_minimum_number_of_peaks.n_required=100000",
        "--report",
        "--json",
        capsys=capsys,
    )

    assert exit_code == 0
    # report is saved next to the output file, replacing the spectra extension
    report_file = str(out.with_suffix("")) + "_processing_report.json"
    assert payload["report_file"] == report_file
    assert os.path.exists(report_file)

    steps = payload["report"]["steps"]
    # one step per executed filter
    assert [s["filter"] for s in steps] == filter_names(payload)
    for step in steps:
        assert set(step) == {
            "filter",
            "input_spectra",
            "output_spectra",
            "removed_spectra",
            "changed_metadata",
            "changed_fragments",
        }

    # the removed-all filter step must reflect the removals
    removed_step = next(s for s in steps if s["filter"] == "require_minimum_number_of_peaks")
    assert removed_step["removed_spectra"] == N_MGF
    assert removed_step["output_spectra"] == 0

    # the report file on disk matches the payload
    with open(report_file) as fh:
        on_disk = json.load(fh)
    assert on_disk["pipeline"] == payload["pipeline"]
    assert on_disk["n_spectra_out"] == payload["n_spectra_out"]
    assert on_disk["steps"] == steps


def test_filter_run_report_counts_add_up(tmp_path, capsys):
    """For a filter that keeps all spectra, input == output and removed == 0."""
    out = tmp_path / "out.msp"
    exit_code, payload = run_cli(
        "filter", "run", MGF_FILE, str(out), "--report", "--json", capsys=capsys
    )
    assert exit_code == 0
    first = payload["report"]["steps"][0]
    assert first["input_spectra"] == N_MGF
    assert first["removed_spectra"] == 0
    assert first["output_spectra"] == N_MGF


# -- error handling ---------------------------------------------------------


def test_filter_run_missing_input(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.mgf"
    out = tmp_path / "out.msp"

    exit_code, payload = run_cli(
        "filter", "run", str(missing), str(out), "--json", capsys=capsys
    )

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "file_not_found"
    assert payload["input_file"] == str(missing)


def test_filter_run_unsupported_input_extension(tmp_path, capsys):
    fake = tmp_path / "spectra.txt"
    fake.write_text("not a spectra file\n", encoding="utf-8")
    out = tmp_path / "out.msp"

    exit_code, payload = run_cli(
        "filter", "run", str(fake), str(out), "--json", capsys=capsys
    )

    assert exit_code == 1
    assert payload["error"] == "unsupported_input_format"
    assert set(payload["valid_values"]) == set(INPUT_FORMATS)


def test_filter_run_unsupported_output_extension(tmp_path, capsys):
    out = tmp_path / "out.xyz"

    exit_code, payload = run_cli(
        "filter", "run", MGF_FILE, str(out), "--json", capsys=capsys
    )

    assert exit_code == 1
    assert payload["error"] == "unsupported_output_format"
    assert set(payload["valid_values"]) == set(OUTPUT_FORMATS)


def test_filter_run_output_exists(tmp_path, capsys):
    out = tmp_path / "out.msp"
    out.write_text("existing\n", encoding="utf-8")

    exit_code, payload = run_cli(
        "filter", "run", MGF_FILE, str(out), "--json", capsys=capsys
    )

    assert exit_code == 1
    assert payload["error"] == "file_exists"


def test_filter_run_empty_input(tmp_path, capsys):
    from matchms.exporting import save_as_mgf

    empty = tmp_path / "empty.mgf"
    save_as_mgf([], str(empty), "matchms", file_mode="w")
    out = tmp_path / "out.msp"

    exit_code, payload = run_cli(
        "filter", "run", str(empty), str(out), "--json", capsys=capsys
    )

    assert exit_code == 1
    assert payload["error"] == "empty_spectra"


def test_filter_run_unknown_filter(tmp_path, capsys):
    out = tmp_path / "out.msp"

    exit_code, payload = run_cli(
        "filter", "run", MGF_FILE, str(out), "--filter", "bogus_filter", "--json",
        capsys=capsys,
    )

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "unknown_value"
    assert "bogus_filter" in payload["message"]
    assert "select_by_mz" in payload["valid_values"]


def test_filter_run_param_on_filter_not_in_pipeline(tmp_path, capsys):
    out = tmp_path / "out.msp"
    # require_minimum_number_of_peaks is not in DEFAULT_FILTERS and not added
    exit_code, payload = run_cli(
        "filter",
        "run",
        MGF_FILE,
        str(out),
        "--param",
        "require_minimum_number_of_peaks.n_required=20",
        "--json",
        capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "unknown_value"
    # the hint should suggest adding the filter
    assert "--filter" in payload["hint"]


def test_filter_run_param_not_accepted_by_filter(tmp_path, capsys):
    out = tmp_path / "out.msp"

    exit_code, payload = run_cli(
        "filter",
        "run",
        MGF_FILE,
        str(out),
        "--filter",
        "select_by_mz",
        "--param",
        "select_by_mz.nonexistent=5",
        "--json",
        capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "select_by_mz.nonexistent"
    assert "mz_from" in payload["valid_values"]
    assert "mz_to" in payload["valid_values"]


def test_filter_run_missing_required_param(tmp_path, capsys):
    out = tmp_path / "out.msp"

    exit_code, payload = run_cli(
        "filter",
        "run",
        MGF_FILE,
        str(out),
        "--filter",
        "repair_smiles_of_salts",
        "--json",
        capsys=capsys,
    )

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "missing_parameter"
    assert "--param" in payload["hint"]


def test_filter_run_malformed_param(tmp_path, capsys):
    out = tmp_path / "out.msp"

    exit_code, payload = run_cli(
        "filter",
        "run",
        MGF_FILE,
        str(out),
        "--param",
        "select_by_mz",  # missing ".name=value"
        "--json",
        capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"


def test_filter_run_requires_positional_args(tmp_path, capsys):
    parser = build_parser()

    # missing output
    with pytest.raises(SystemExit):
        parser.parse_args(["filter", "run", MGF_FILE])

    # both missing
    with pytest.raises(SystemExit):
        parser.parse_args(["filter", "run"])


# -- output styles / append (parity with spectra convert) --------------------


def test_filter_run_append_to_msp(tmp_path, capsys):
    out = tmp_path / "out.msp"

    exit_code, _ = run_cli(
        "filter", "run", MGF_FILE, str(out), "--json", capsys=capsys
    )
    assert exit_code == 0
    first_n = load_n(str(out), ftype="msp")

    # second run appends instead of erroring
    exit_code, payload = run_cli(
        "filter", "run", MGF_FILE, str(out), "--append", "--json", capsys=capsys
    )
    assert exit_code == 0
    assert load_n(str(out), ftype="msp") == first_n + payload["n_spectra_out"]


def test_filter_run_append_unsupported_format(tmp_path, capsys):
    out = tmp_path / "out.json"

    exit_code, payload = run_cli(
        "filter", "run", MGF_FILE, str(out), "--append", "--json", capsys=capsys
    )

    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "append"
