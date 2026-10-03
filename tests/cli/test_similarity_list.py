"""Tests for the `matchms similarity list` CLI command."""

import json
import pytest
from matchms.cli.commands.similarity_list import GROUPS, OTHER_GROUP_NAME
from matchms.cli.main import build_parser, main
from matchms.similarity import __all__ as SIMILARITY_NAMES


def run_cli(*argv, capsys=None):
    """Run the CLI and return (exit_code, parsed_json_stdout)."""
    exit_code = main(list(argv))
    out = capsys.readouterr().out
    payload = json.loads(out)
    return exit_code, payload


def _grouped_names():
    return {name for _, _, names in GROUPS for name in names}


def test_similarity_list_json(capsys):
    exit_code, payload = run_cli("similarity", "list", "--json", capsys=capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["operation"] == "similarity list"
    # every class in matchms.similarity is listed, exactly once
    assert payload["n_similarities"] == len(SIMILARITY_NAMES)
    names = [entry["name"] for entry in payload["similarities"]]
    assert sorted(names) == sorted(SIMILARITY_NAMES)
    assert len(set(names)) == len(names)

    # every entry carries group, name, description, score_fields and methods
    for entry in payload["similarities"]:
        assert set(entry) == {"group", "name", "description", "score_fields", "methods"}
        assert isinstance(entry["group"], str) and entry["group"]
        assert isinstance(entry["name"], str)
        assert isinstance(entry["description"], str) and entry["description"]
        assert entry["score_fields"]
        assert "pair" in entry["methods"]


def test_similarity_list_grouped_like_readme(capsys):
    """Classes are listed in the README groups, in the README member order."""
    exit_code, payload = run_cli("similarity", "list", "--json", capsys=capsys)

    assert exit_code == 0
    by_name = {entry["name"]: entry for entry in payload["similarities"]}

    expected_order = []
    for group_name, _, class_names in GROUPS:
        for name in class_names:
            expected_order.append(name)
            assert by_name[name]["group"] == group_name
    # README-grouped classes come first, in the documented order
    listed = [entry["name"] for entry in payload["similarities"]]
    assert listed[: len(expected_order)] == expected_order

    # the Cosine group has the specialized implementations, as in the README
    assert [
        entry["name"] for entry in payload["similarities"] if entry["group"] == "Cosine"
    ] == ["Cosine", "CosineGreedy", "CosineHungarian", "CosineLinear", "CosineFlash", "CosineBlink"]
    assert [
        entry["name"] for entry in payload["similarities"] if entry["group"] == "Modified cosine"
    ] == ["ModifiedCosine", "ModifiedCosineGreedy", "ModifiedCosineHungarian", "ModifiedCosineLinear"]
    assert [
        entry["name"] for entry in payload["similarities"] if entry["group"] == "Spectral entropy"
    ] == ["Entropy", "EntropyGreedy", "EntropyFlash"]


def test_similarity_list_ungrouped_classes_in_other(capsys):
    """Classes that are not in any README group are listed under 'Other'."""
    exit_code, payload = run_cli("similarity", "list", "--json", capsys=capsys)

    assert exit_code == 0
    by_name = {entry["name"]: entry for entry in payload["similarities"]}
    other = [name for name in SIMILARITY_NAMES if name not in _grouped_names()]
    for name in other:
        assert by_name[name]["group"] == OTHER_GROUP_NAME
    for entry in payload["similarities"]:
        if entry["group"] == OTHER_GROUP_NAME:
            assert entry["name"] in other


def test_similarity_list_methods_reflect_capability(capsys):
    """The reported methods match what the classes actually support."""
    exit_code, payload = run_cli("similarity", "list", "--json", capsys=capsys)

    assert exit_code == 0
    by_name = {entry["name"]: entry for entry in payload["similarities"]}

    # Cosine supports the indexed search workflow, CosineGreedy does not
    assert "build_index" in by_name["Cosine"]["methods"]
    assert "search" in by_name["Cosine"]["methods"]
    assert "build_index" not in by_name["CosineGreedy"]["methods"]
    assert "search" not in by_name["CosineGreedy"]["methods"]

    # score fields come from the class
    assert by_name["Cosine"]["score_fields"] == ["score", "matches"]
    assert by_name["MetadataMatch"]["score_fields"] == ["score"]
    assert by_name["MetadataMatch"]["description"].startswith("Return True if metadata entries")


def test_similarity_list_description_is_first_line(capsys):
    """The description should be the leading line of the class docstring."""
    exit_code, payload = run_cli("similarity", "list", "--json", capsys=capsys)
    assert exit_code == 0
    by_name = {entry["name"]: entry for entry in payload["similarities"]}

    assert by_name["Cosine"]["description"].startswith("Compare mass spectra using cosine similarity.")
    assert by_name["EntropySearch"]["description"].startswith(
        "Search-optimized fragment spectral entropy similarity."
    )


def test_similarity_list_table_output(capsys):
    exit_code = main(["similarity", "list", "--table"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Available similarity measures" in out
    for column in ("group", "name", "description", "methods"):
        assert column in out
    assert "Cosine" in out
    assert "CosineGreedy" in out
    assert "ModifiedCosine" in out
    # forced --table must not emit JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_similarity_list_requires_no_action(capsys):
    parser = build_parser()
    # `similarity` with no subcommand has no runnable func
    args = parser.parse_args(["similarity"])
    assert not hasattr(args, "func")
    # `similarity` with an unknown subcommand exits with code 2
    with pytest.raises(SystemExit):
        parser.parse_args(["similarity", "bogus"])
