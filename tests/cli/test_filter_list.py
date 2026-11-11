import json
import pytest
from matchms.cli.main import main
from matchms.filtering.filter_order import ALL_FILTERS


def run_cli(*argv, capsys=None):
    """Run the CLI and return (exit_code, parsed_json_stdout)."""
    exit_code = main(list(argv))
    out = capsys.readouterr().out
    payload = json.loads(out)
    return exit_code, payload


def test_filter_list_json(capsys):
    exit_code, payload = run_cli("filter", "list", "--json", capsys=capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["operation"] == "filter list"
    assert payload["n_filters"] == len(ALL_FILTERS)
    assert len(payload["filters"]) == len(ALL_FILTERS)

    # every entry carries order, name and description
    for entry in payload["filters"]:
        assert set(entry) == {"order", "name", "description"}
        assert isinstance(entry["order"], int)
        assert isinstance(entry["name"], str)
        assert isinstance(entry["description"], str) and entry["description"]

    # order is contiguous 0..n-1 and matches ALL_FILTERS
    assert [e["order"] for e in payload["filters"]] == list(range(len(ALL_FILTERS)))
    assert [e["name"] for e in payload["filters"]] == [f.__name__ for f in ALL_FILTERS]


def test_filter_list_order_matches_all_filters(capsys):
    """The listed names must be exactly the ALL_FILTERS names, in the same order."""
    exit_code, payload = run_cli("filter", "list", "--json", capsys=capsys)

    assert exit_code == 0
    assert [e["name"] for e in payload["filters"]] == [f.__name__ for f in ALL_FILTERS]


def test_filter_list_description_is_first_line(capsys):
    """The description should be the leading sentence of the filter docstring."""
    exit_code, payload = run_cli("filter", "list", "--json", capsys=capsys)
    assert exit_code == 0
    by_name = {e["name"]: e for e in payload["filters"]}

    # make_charge_int's docstring starts with "Convert charge field to integer..."
    assert by_name["make_charge_int"]["description"].startswith("Convert charge field to integer")
    # select_by_mz's docstring starts with "Keep only peaks between mz_from and mz_to."
    assert by_name["select_by_mz"]["description"].startswith("Keep only peaks between mz_from and mz_to.")


def test_filter_list_table_output(capsys):
    exit_code = main(["filter", "list", "--table"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Available filters" in out
    assert "order" in out
    assert "name" in out
    assert "description" in out
    assert "make_charge_int" in out
    # forced --table must not emit JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)
