import json
import pytest
from matchms.cli.main import build_parser, main
from matchms.filtering.default_pipelines import FILTER_SETS_BY_NAME


def run_cli(*argv, capsys=None):
    """Run the CLI and return (exit_code, parsed_json_stdout)."""
    exit_code = main(list(argv))
    out = capsys.readouterr().out
    payload = json.loads(out)
    return exit_code, payload


# -- overview (no pipeline name) -------------------------------------------


def test_filter_pipelines_overview_json(capsys):
    exit_code, payload = run_cli("filter", "pipelines", "--json", capsys=capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["operation"] == "filter pipelines"
    assert payload["default_pipeline"] == "DEFAULT_FILTERS"
    assert payload["n_pipelines"] == len(FILTER_SETS_BY_NAME)
    assert len(payload["pipelines"]) == len(FILTER_SETS_BY_NAME)

    # every entry carries name, description, n_filters and the default flag
    for entry in payload["pipelines"]:
        assert set(entry) == {"name", "description", "n_filters", "is_default"}
        assert isinstance(entry["name"], str) and entry["name"]
        assert isinstance(entry["description"], str) and entry["description"]
        assert isinstance(entry["n_filters"], int)
        assert isinstance(entry["is_default"], bool)
        # the overview must not expand the filter steps
        assert "filters" not in entry


def test_filter_pipelines_overview_names_and_order_match_definition(capsys):
    """The listed names must be exactly the FILTER_SETS_BY_NAME keys, in the same order."""
    exit_code, payload = run_cli("filter", "pipelines", "--json", capsys=capsys)

    assert exit_code == 0
    assert [e["name"] for e in payload["pipelines"]] == list(FILTER_SETS_BY_NAME)


def test_filter_pipelines_overview_is_default_flag(capsys):
    exit_code, payload = run_cli("filter", "pipelines", "--json", capsys=capsys)

    assert exit_code == 0
    defaults = [e["name"] for e in payload["pipelines"] if e["is_default"]]
    assert defaults == ["DEFAULT_FILTERS"]
    assert payload["default_pipeline"] == defaults[0]


def test_filter_pipelines_overview_n_filters(capsys):
    exit_code, payload = run_cli("filter", "pipelines", "--json", capsys=capsys)

    assert exit_code == 0
    by_name = {e["name"]: e for e in payload["pipelines"]}
    for name, filter_set in FILTER_SETS_BY_NAME.items():
        assert by_name[name]["n_filters"] == len(filter_set)


def test_filter_pipelines_overview_table_output(capsys):
    exit_code = main(["filter", "pipelines", "--table"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Available pipelines" in out
    assert "name" in out
    assert "n_filters" in out
    assert "description" in out
    assert "DEFAULT_FILTERS" in out
    assert "MS2DEEPSCORE_TRAINING" in out
    # the overview must not list the individual filters of each pipeline
    assert "Filters per pipeline" not in out
    assert "make_charge_int" not in out
    # forced --table must not emit JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


# -- detail (pipeline name as argument) ------------------------------------


def test_filter_pipelines_detail_json(capsys):
    exit_code, payload = run_cli("filter", "pipelines", "HARMONIZE_METADATA_FIELD_NAMES", "--json", capsys=capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["operation"] == "filter pipelines HARMONIZE_METADATA_FIELD_NAMES"
    assert payload["name"] == "HARMONIZE_METADATA_FIELD_NAMES"
    assert isinstance(payload["description"], str) and payload["description"]
    assert payload["is_default"] is False
    assert payload["n_filters"] == len(payload["filters"])

    # every step carries a name (and parameters only when the pipeline sets them)
    for step in payload["filters"]:
        assert "name" in step
        assert isinstance(step["name"], str) and step["name"]
        assert ("parameters" in step) == bool(step.get("parameters"))


def test_filter_pipelines_detail_filters_match_filter_set(capsys):
    """The detail filters must mirror the pipeline's filter set exactly."""
    exit_code, payload = run_cli("filter", "pipelines", "REQUIRE_COMPLETE_METADATA", "--json", capsys=capsys)

    assert exit_code == 0
    filter_set = FILTER_SETS_BY_NAME["REQUIRE_COMPLETE_METADATA"]
    steps = payload["filters"]
    assert len(steps) == len(filter_set)
    for step, entry in zip(steps, filter_set, strict=True):
        if isinstance(entry, (tuple, list)):
            assert step["name"] == entry[0].__name__
            assert step["parameters"] == dict(entry[1])
        else:
            assert step["name"] == entry.__name__
            assert "parameters" not in step


def test_filter_pipelines_detail_parameters(capsys):
    """Pipelines with parametrized filters must report those parameters."""
    exit_code, payload = run_cli("filter", "pipelines", "REQUIRE_COMPLETE_METADATA", "--json", capsys=capsys)

    assert exit_code == 0
    parametrized = [step for step in payload["filters"] if step.get("parameters")]
    assert parametrized == [{"name": "require_correct_ionmode", "parameters": {"ion_mode_to_keep": "both"}}]


def test_filter_pipelines_detail_is_default(capsys):
    exit_code, payload = run_cli("filter", "pipelines", "DEFAULT_FILTERS", "--json", capsys=capsys)

    assert exit_code == 0
    assert payload["is_default"] is True
    assert payload["n_filters"] == len(FILTER_SETS_BY_NAME["DEFAULT_FILTERS"])


def test_filter_pipelines_detail_table_output(capsys):
    exit_code = main(["filter", "pipelines", "HARMONIZE_METADATA_FIELD_NAMES", "--table"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Pipeline: HARMONIZE_METADATA_FIELD_NAMES" in out
    assert "Filters (in execution order):" in out
    assert "  HARMONIZE_METADATA_FIELD_NAMES:" in out
    for name in (
        "make_charge_int",
        "add_compound_name",
        "interpret_pepmass",
        "add_precursor_mz",
        "add_retention_time",
        "add_retention_index",
    ):
        assert f"    - {name}" in out
    # the overview heading must not appear in the detail output
    assert "Available pipelines" not in out
    # forced --table must not emit JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_filter_pipelines_detail_table_shows_parameters(capsys):
    exit_code = main(["filter", "pipelines", "REQUIRE_COMPLETE_METADATA", "--table"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "require_correct_ionmode (ion_mode_to_keep=both)" in out


def test_filter_pipelines_detail_all_names(capsys):
    """Every defined pipeline name can be looked up by its detail command."""
    exit_code, payload = run_cli("filter", "pipelines", "--json", capsys=capsys)
    assert exit_code == 0
    for name in [e["name"] for e in payload["pipelines"]]:
        exit_code, detail = run_cli("filter", "pipelines", name, "--json", capsys=capsys)
        assert exit_code == 0
        assert detail["name"] == name
        assert detail["n_filters"] == len(FILTER_SETS_BY_NAME[name])


# -- errors and parser ------------------------------------------------------


def test_filter_pipelines_unknown_name_error(capsys):
    exit_code, payload = run_cli("filter", "pipelines", "NOT_A_PIPELINE", "--json", capsys=capsys)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "unknown_value"
    assert payload["operation"] == "filter pipelines NOT_A_PIPELINE"
    assert payload["parameter"] == "pipeline_name"
    assert "NOT_A_PIPELINE" in payload["message"]
    # the valid pipeline names are offered as valid_values
    assert payload["valid_values"] == sorted(FILTER_SETS_BY_NAME)


def test_filter_pipelines_requires_no_action(capsys):
    parser = build_parser()
    args = parser.parse_args(["filter", "pipelines"])
    assert hasattr(args, "func")
    assert args.pipeline_name is None
    # an unknown subcommand still exits with code 2
    with pytest.raises(SystemExit):
        parser.parse_args(["filter", "bogus"])
