"""Tests for the `matchms similarity info` CLI command."""

import json
import pytest
from matchms.cli.commands.similarity_list import GROUPS, OTHER_GROUP_NAME
from matchms.cli.main import build_parser, main
from matchms.similarity import (
    Cosine,
    CosineBlink,
    CosineGreedy,
    FingerprintSimilarity,
    MetadataMatch,
    get_similarity_function_by_name,
)
from matchms.similarity import (
    __all__ as SIMILARITY_NAMES,
)


def run_cli(*argv, capsys=None):
    """Run the CLI and return (exit_code, parsed_json_stdout)."""
    exit_code = main(list(argv))
    out = capsys.readouterr().out
    payload = json.loads(out)
    return exit_code, payload


def _group_of(name):
    for group_name, _, class_names in GROUPS:
        if name in class_names:
            return group_name
    return OTHER_GROUP_NAME


def test_similarity_info_json(capsys):
    exit_code, payload = run_cli("similarity", "info", "CosineGreedy", "--json", capsys=capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["operation"] == "similarity info"
    assert payload["name"] == "CosineGreedy"
    assert payload["group"] == _group_of("CosineGreedy") == "Cosine"
    assert payload["description"].startswith("Calculate 'cosine similarity score'")
    assert payload["score_fields"] == ["score", "matches"]
    assert payload["is_commutative"] is True
    assert set(payload["methods"]) == {"pair", "matrix", "sparse_matrix"}
    assert isinstance(payload["docstring"], str) and payload["docstring"]

    # parameters are reported with type, required flag and default
    params = {p["name"]: p for p in payload["parameters"]}
    assert set(params) == {
        "tolerance", "mz_power", "intensity_power",
        "noise_cutoff", "remove_precursor", "offset_to_precursor",
    }
    assert params["tolerance"]["required"] is False
    assert params["tolerance"]["default"] == 0.01
    assert params["tolerance"]["type"] == "float"
    assert payload["required"] == []


def test_similarity_info_methods_reflect_capability(capsys):
    """Indexed search methods are only reported when the class supports them."""
    for name, indexed in (("Cosine", True), ("CosineGreedy", False), ("EntropySearch", True)):
        exit_code, payload = run_cli("similarity", "info", name, "--json", capsys=capsys)
        assert exit_code == 0
        assert ("build_index" in payload["methods"]) is indexed
        assert ("search" in payload["methods"]) is indexed


def test_similarity_info_required_parameter(capsys):
    """FingerprintSimilarity has required constructor parameters."""
    exit_code, payload = run_cli(
        "similarity", "info", "FingerprintSimilarity", "--json", capsys=capsys
    )

    assert exit_code == 0
    assert "fingerprint_generator" in payload["required"]
    params = {p["name"]: p for p in payload["parameters"]}
    assert params["fingerprint_generator"]["required"] is True
    assert "default" not in params["fingerprint_generator"]
    assert params["similarity_measure"]["required"] is False
    assert params["similarity_measure"]["default"] == "tanimoto"


def test_similarity_info_parameters_match_signature(capsys):
    """The reported parameters must match the class constructor signature."""
    import inspect

    for name in ("Cosine", "CosineBlink", "MetadataMatch", "PrecursorMzMatch", "CosineGreedy"):
        exit_code, payload = run_cli("similarity", "info", name, "--json", capsys=capsys)
        assert exit_code == 0, name
        reported = {p["name"] for p in payload["parameters"]}
        expected = set(inspect.signature(get_similarity_function_by_name(name)).parameters)
        assert reported == expected


def test_similarity_info_parameter_description(capsys):
    """Parameter descriptions come from the docstring when available."""
    # Cosine documents its parameters in the class docstring
    exit_code, payload = run_cli("similarity", "info", "Cosine", "--json", capsys=capsys)
    assert exit_code == 0
    params = {p["name"]: p for p in payload["parameters"]}
    assert params["tolerance"]["description"].startswith(
        "Maximum difference between two fragment m/z values"
    )
    assert "intensity" in params["noise_cutoff"]["description"].lower()


def test_similarity_info_group_assignment(capsys):
    exit_code, payload = run_cli("similarity", "info", "ModifiedCosine", "--json", capsys=capsys)
    assert exit_code == 0
    assert payload["group"] == "Modified cosine"

    exit_code, payload = run_cli("similarity", "info", "ParentMassMatch", "--json", capsys=capsys)
    assert exit_code == 0
    assert payload["group"] == "Precursor or parent mass"


def test_similarity_info_table_output(capsys):
    exit_code = main(["similarity", "info", "CosineGreedy", "--table"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Similarity: CosineGreedy" in out
    assert "group: Cosine" in out
    assert "score fields: score, matches" in out
    assert "What it does:" in out
    assert "Constructor parameters:" in out
    assert "tolerance" in out
    assert "noise_cutoff" in out
    # forced --table must not emit JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_similarity_info_unknown_similarity(capsys):
    exit_code, payload = run_cli("similarity", "info", "not_a_similarity", "--json", capsys=capsys)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "unknown_value"
    assert "not_a_similarity" in payload["message"]
    assert "similarity_name" == payload["parameter"]
    # the list of valid similarities should include real names
    assert "Cosine" in payload["valid_values"]
    assert "ModifiedCosine" in payload["valid_values"]
    assert "not_a_similarity" not in payload["valid_values"]


def test_similarity_info_requires_similarity_name(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["similarity", "info"])

    with pytest.raises(SystemExit):
        parser.parse_args(["similarity", "info", "Cosine", "Cosine"])


def test_similarity_info_all_similarities_resolvable(capsys):
    """Every class in matchms.similarity must be resolvable via `similarity info`."""
    for name in SIMILARITY_NAMES:
        exit_code, payload = run_cli("similarity", "info", name, "--json", capsys=capsys)
        assert exit_code == 0, name
        assert payload["name"] == name
        assert payload["group"] == _group_of(name)
        assert payload["docstring"]
        assert [p["name"] for p in payload["parameters"]]

    # the class under test is importable (also documents the public names used here)
    assert Cosine is get_similarity_function_by_name("Cosine")
    assert CosineBlink is get_similarity_function_by_name("CosineBlink")
    assert CosineGreedy is get_similarity_function_by_name("CosineGreedy")
    assert FingerprintSimilarity is get_similarity_function_by_name("FingerprintSimilarity")
    assert MetadataMatch is get_similarity_function_by_name("MetadataMatch")
