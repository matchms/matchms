import json
import pytest
from matchms.cli.main import build_parser, main
from matchms.filtering.filter_order import ALL_FILTERS, FILTER_FUNCTION_NAMES


def run_cli(*argv, capsys=None):
    """Run the CLI and return (exit_code, parsed_json_stdout)."""
    exit_code = main(list(argv))
    out = capsys.readouterr().out
    payload = json.loads(out)
    return exit_code, payload


def test_filter_info_json(capsys):
    exit_code, payload = run_cli("filter", "info", "select_by_mz", "--json", capsys=capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["operation"] == "filter info"
    assert payload["name"] == "select_by_mz"
    # order is the position in ALL_FILTERS
    assert payload["order"] == ALL_FILTERS.index(FILTER_FUNCTION_NAMES["select_by_mz"])
    assert payload["collection_supported"] is True
    assert isinstance(payload["docstring"], str) and payload["docstring"]
    assert payload["description"].startswith("Keep only peaks between mz_from and mz_to.")
    # mz_from / mz_to are optional parameters with defaults
    params = {p["name"]: p for p in payload["parameters"]}
    assert set(params) == {"mz_from", "mz_to"}
    assert params["mz_from"]["required"] is False
    assert params["mz_from"]["default"] == 0.0
    assert params["mz_to"]["default"] == 1000.0
    # the internal clone flag must not be exposed to users
    assert "clone" not in params


def test_filter_info_required_parameter(capsys):
    """repair_smiles_of_salts has a required mass_tolerance parameter."""
    exit_code, payload = run_cli("filter", "info", "repair_smiles_of_salts", "--json", capsys=capsys)

    assert exit_code == 0
    assert payload["name"] == "repair_smiles_of_salts"
    assert payload["required"] == ["mass_tolerance"]
    params = {p["name"]: p for p in payload["parameters"]}
    assert params["mass_tolerance"]["required"] is True
    assert "default" not in params["mass_tolerance"]
    # the required parameter is listed even though the docstring does not
    # document it (signature is the source of truth for the parameter list)
    assert params["mass_tolerance"]["type"] == "float"


def test_filter_info_parameters_match_signature(capsys):
    """The reported parameters must match the filter's actual signature."""

    from matchms.cli.introspection import filter_parameter_names

    for name in ("select_by_mz", "harmonize_missing_entries", "require_minimum_number_of_peaks"):
        exit_code, payload = run_cli("filter", "info", name, "--json", capsys=capsys)
        assert exit_code == 0
        reported = {p["name"] for p in payload["parameters"]}
        expected = set(filter_parameter_names(FILTER_FUNCTION_NAMES[name]))
        assert reported == expected


def test_filter_info_table_output(capsys):
    exit_code = main(["filter", "info", "select_by_mz", "--table"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Filter: select_by_mz" in out
    assert "What it does:" in out
    assert "Parameters:" in out
    assert "mz_from" in out
    assert "mz_to" in out
    # forced --table must not emit JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_filter_info_unknown_filter(capsys):
    exit_code, payload = run_cli("filter", "info", "not_a_filter", "--json", capsys=capsys)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "unknown_value"
    assert "not_a_filter" in payload["message"]
    # the list of valid filters should include real names
    assert "select_by_mz" in payload["valid_values"]
    assert "make_charge_int" in payload["valid_values"]


def test_filter_info_requires_filter_name(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["filter", "info"])

    with pytest.raises(SystemExit):
        parser.parse_args(["filter", "info", "select_by_mz", "select_by_mz"])


def test_filter_info_all_filters_resolvable(capsys):
    """Every filter in ALL_FILTERS must be resolvable via `filter info`."""
    for func in ALL_FILTERS:
        exit_code, payload = run_cli("filter", "info", func.__name__, "--json", capsys=capsys)
        assert exit_code == 0, func.__name__
        assert payload["name"] == func.__name__
