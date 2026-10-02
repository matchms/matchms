"""Unit tests for the CLI helpers used by the filter commands.

Covers the pure helpers in :mod:`matchms.cli.params`
(:func:`parse_scoped_params`, :func:`convert_value`) and the filter
introspection helpers in :mod:`matchms.cli.introspection`.
"""

import pytest
from matchms.cli.errors import CliError
from matchms.cli.introspection import (
    filter_accepts_parameter,
    filter_description,
    filter_parameter_names,
    filter_signature,
)
from matchms.cli.params import convert_value, parse_scoped_params
from matchms.filtering.filter_order import FILTER_FUNCTION_NAMES


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
