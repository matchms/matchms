"""Unit tests for the CLI helpers used by the filter and similarity commands.

Covers the pure helpers in :mod:`matchms.cli.params`
(:func:`parse_scoped_params`, :func:`convert_value`) and the introspection
helpers in :mod:`matchms.cli.introspection` (filter signatures and
similarity signatures, including bare-name parameter docs).
"""

import pytest
from matchms.cli.errors import CliError
from matchms.cli.introspection import (
    extract_param_docs,
    filter_accepts_parameter,
    filter_description,
    filter_parameter_names,
    filter_signature,
    short_description,
    signature_parameters,
    similarity_signature,
)
from matchms.cli.params import convert_value, parse_scoped_params
from matchms.filtering.filter_order import FILTER_FUNCTION_NAMES
from matchms.similarity import (
    Cosine,
    CosineGreedy,
    CosineLinear,
    EntropySearch,
    get_similarity_function_by_name,
)


# -- parse_scoped_params ----------------------------------------------------


def test_parse_scoped_params_empty():
    assert parse_scoped_params(None) == {}
    assert parse_scoped_params([]) == {}


def test_parse_scoped_params_single():
    assert parse_scoped_params(["select_by_mz.mz_from=10.0"]) == {"select_by_mz": {"mz_from": 10.0}}


def test_parse_scoped_params_multiple_filters():
    result = parse_scoped_params(
        [
            "select_by_mz.mz_from=10",
            "select_by_mz.mz_to=500",
            "require_minimum_number_of_peaks.n_required=5",
        ]
    )
    assert result == {
        "select_by_mz": {"mz_from": 10, "mz_to": 500},
        "require_minimum_number_of_peaks": {"n_required": 5},
    }


@pytest.mark.parametrize(
    "raw",
    ["noequals", "filter_only", ".mz_from=1", "select_by_mz.=1", "select_by_mz.mz_from"],
)
def test_parse_scoped_params_malformed(raw):
    with pytest.raises(CliError) as exc:
        parse_scoped_params([raw])
    assert exc.value.code == "invalid_parameter"


def test_parse_scoped_params_types():
    result = parse_scoped_params(
        [
            "f.a=1",  # int
            "f.b=1.5",  # float
            "f.c=true",  # bool
            "f.d=false",  # bool
            "f.e=hello",  # str
            'f.g=["x", "y"]',  # list
        ]
    )
    assert result["f"] == {"a": 1, "b": 1.5, "c": True, "d": False, "e": "hello", "g": ["x", "y"]}


# -- convert_value ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("False", False),
        ("none", None),
        ("42", 42),
        ("3.14", 3.14),
        ("-7", -7),
        ("hello", "hello"),
        ('["a", "b"]', ["a", "b"]),
        ('{"k": "v"}', {"k": "v"}),
    ],
)
def test_convert_value(raw, expected):
    assert convert_value("some.param", raw) == expected


# -- introspection helpers --------------------------------------------------


def test_filter_parameter_names_excludes_input_and_clone():
    func = FILTER_FUNCTION_NAMES["select_by_mz"]
    names = filter_parameter_names(func)
    assert "spectrum_in" not in names
    assert "clone" not in names
    assert set(names) == {"mz_from", "mz_to"}


def test_filter_parameter_names_for_metadata_filter():
    func = FILTER_FUNCTION_NAMES["require_minimum_number_of_peaks"]
    assert set(filter_parameter_names(func)) == {"n_required", "ratio_required"}


def test_filter_accepts_parameter():
    func = FILTER_FUNCTION_NAMES["select_by_mz"]
    assert filter_accepts_parameter(func, "mz_from") is True
    assert filter_accepts_parameter(func, "mz_to") is True
    assert filter_accepts_parameter(func, "clone") is False
    assert filter_accepts_parameter(func, "not_a_param") is False


def test_filter_signature_shape():
    func = FILTER_FUNCTION_NAMES["select_by_mz"]
    info = filter_signature(func)
    assert set(info) == {"signature", "required", "collection_supported", "docstring", "param_docs"}
    # the raw signature is reported faithfully; the spectrum input is excluded
    # (it is managed by the processor, not the user)
    assert "spectrum_in" not in info["signature"]["parameters"]
    assert "mz_from" in info["signature"]["parameters"]
    assert "mz_to" in info["signature"]["parameters"]
    assert info["required"] == []


def test_filter_signature_required_flag():
    func = FILTER_FUNCTION_NAMES["repair_smiles_of_salts"]
    info = filter_signature(func)
    assert info["required"] == ["mass_tolerance"]


def test_filter_description_no_params():
    func = FILTER_FUNCTION_NAMES["make_charge_int"]
    assert filter_description(func) == {"name": "make_charge_int"}


def test_filter_description_with_params():
    func = FILTER_FUNCTION_NAMES["select_by_mz"]
    assert filter_description(func, {"mz_from": 10}) == {
        "name": "select_by_mz",
        "parameters": {"mz_from": 10},
    }


# -- extract_param_docs (bare parameter names) ------------------------------


def test_extract_param_docs_bare_names():
    """Similarity docstrings use bare parameter names (no ``name:`` prefix)."""
    docs = extract_param_docs(EntropySearch.__doc__)
    assert "tolerance" in docs
    assert docs["tolerance"].startswith("Maximum absolute fragment m/z difference")
    assert "max_tolerance" in docs
    assert "peak_separation" in docs


def test_extract_param_docs_typed_names():
    """The classic ``name: type`` style still works."""
    docs = extract_param_docs(CosineGreedy.__init__.__doc__)
    assert "tolerance" in docs
    assert docs["tolerance"].startswith("Peaks will be considered a match")


def test_extract_param_docs_class_and_init_fallback():
    """Cosine documents its parameters in the class docstring only."""
    assert "tolerance" in extract_param_docs(Cosine.__doc__)
    assert "tolerance" not in extract_param_docs(Cosine.__init__.__doc__)


# -- similarity_signature ---------------------------------------------------


def test_similarity_signature_shape():
    info = similarity_signature(CosineGreedy)
    assert set(info) == {
        "signature",
        "required",
        "param_docs",
        "docstring",
        "score_fields",
        "is_commutative",
        "methods",
    }
    assert info["score_fields"] == ["score", "matches"]
    assert info["is_commutative"] is True
    assert info["docstring"].startswith("Calculate 'cosine similarity score'")
    assert set(info["signature"]["parameters"]) == {
        "tolerance",
        "mz_power",
        "intensity_power",
        "noise_cutoff",
        "remove_precursor",
        "offset_to_precursor",
    }
    assert info["required"] == []


def test_similarity_signature_param_docs():
    """Parameter docs come from the init docstring when present."""
    info = similarity_signature(CosineLinear)
    assert "tolerance" in info["param_docs"]
    assert info["param_docs"]["tolerance"].startswith("Peaks will be considered a match")
    # Cosine documents in the class docstring instead
    info = similarity_signature(Cosine)
    assert "tolerance" in info["param_docs"]
    assert info["param_docs"]["tolerance"].startswith("Maximum difference between two fragment m/z")


def test_similarity_signature_methods_reflect_capability():
    """Inherited implementations count; a non-implemented sparse_matrix does not."""
    assert similarity_signature(Cosine)["methods"] == ["pair", "matrix", "build_index", "search"]
    assert similarity_signature(CosineGreedy)["methods"] == ["pair", "matrix", "sparse_matrix"]
    assert similarity_signature(EntropySearch)["methods"] == ["pair", "matrix", "build_index", "search"]


def test_similarity_signature_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown similarity function"):
        get_similarity_function_by_name("not_a_similarity")


# -- short_description ------------------------------------------------------


def test_short_description_prefers_first_line():
    assert short_description("Do a thing.\nMore detail.") == "Do a thing."


def test_short_description_cleans_and_handles_empty():
    # raw (un-cleaned) docstring: leading/trailing blank lines are dropped
    assert short_description("  First line.  \n\n  Second line.\n\n") == "First line."
    assert short_description("") == ""
    assert short_description(None) == ""
    # no sentence end -> the (first-line) text is returned as-is
    assert short_description("No sentence end") == "No sentence end"


# -- signature_parameters ---------------------------------------------------


def test_signature_parameters_basic():
    info = filter_signature(FILTER_FUNCTION_NAMES["select_by_mz"])
    # ``filter info`` skips the processor-managed ``clone`` flag
    params = {p["name"]: p for p in signature_parameters(info, skip=("clone",))}
    assert set(params) == {"mz_from", "mz_to"}
    assert params["mz_from"]["required"] is False
    assert params["mz_from"]["default"] == 0.0
    assert "description" in params["mz_from"]


def test_signature_parameters_skip():
    """Named parameters (e.g. ``clone``) are dropped."""
    info = filter_signature(FILTER_FUNCTION_NAMES["select_by_mz"])
    # without skip, ``clone`` (a documented, optional param) would be present
    all_names = {p["name"] for p in signature_parameters(info)}
    skipped_names = {p["name"] for p in signature_parameters(info, skip=("clone",))}
    assert "clone" in all_names
    assert "clone" not in skipped_names


def test_signature_parameters_similarity():
    info = similarity_signature(CosineGreedy)
    params = {p["name"]: p for p in signature_parameters(info)}
    assert set(params) == {
        "tolerance",
        "mz_power",
        "intensity_power",
        "noise_cutoff",
        "remove_precursor",
        "offset_to_precursor",
    }
    assert params["tolerance"]["default"] == 0.01
    assert "description" in params["tolerance"]


def test_signature_parameters_required_listing():
    info = similarity_signature(get_similarity_function_by_name("FingerprintSimilarity"))
    parameters = signature_parameters(info)
    required = [p["name"] for p in parameters if p["required"]]
    assert "fingerprint_generator" in required
    # required entries carry no ``default`` key
    by_name = {p["name"]: p for p in parameters}
    assert "default" not in by_name["fingerprint_generator"]
