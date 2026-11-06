"""Tests for the unified, agent-friendly CLI error protocol.

These tests lock in the acceptance criteria of the error-handling spec:

* every failure (parser, domain, unexpected) emits exactly one JSON document
  on stdout with the stable envelope
  ``{schema_version, ok, error{...}, output_produced{...}}``;
* parser errors are structured (never printed / exit by argparse) and return
  exit code 2;
* ``error.code`` is the semantic class an agent can branch on;
* no traceback is emitted by default.

The specific invocations from the spec are covered here end to end.
"""

import json
import pytest
from matchms.cli.errors import EXIT_INTERNAL, EXIT_USAGE, ArgparseError, CliError
from matchms.cli.main import build_parser, main


TOP_LEVEL_KEYS = {"schema_version", "ok", "error", "output_produced"}


def run_cli(*argv, capsys):
    """Run the CLI, returning (exit_code, raw_stdout, parsed_payload_or_None)."""
    exit_code = main(list(argv))
    out = capsys.readouterr().out
    payload = json.loads(out) if out.strip() else None
    return exit_code, out, payload


def assert_envelope(payload, expected_code):
    """Assert *payload* is a well-formed error envelope with the given code."""
    assert set(payload) == TOP_LEVEL_KEYS, set(payload)
    assert payload["ok"] is False
    assert isinstance(payload["schema_version"], str)
    error = payload["error"]
    assert error["code"] == expected_code
    assert isinstance(error["message"], str) and error["message"]
    assert "Traceback" not in error["message"]
    assert payload["output_produced"] == {"any": False, "files": []}


# -- the spec's acceptance invocations ---------------------------------------
# Each entry: (argv, expected error code). All must be a single JSON error on
# stdout and exit with code 2 (usage / argument validation).
ACCEPTANCE_CASES = [
    ([], "missing_command"),
    (["unknown-command"], "invalid_value"),
    (["filter"], "missing_subcommand"),
    (["filter", "unknown-action"], "invalid_value"),
    (["similarity", "matrix"], "missing_argument"),
    (["similarity", "matrix", "file.mgf", "--unknown"], "invalid_argument"),
    (["similarity", "matrix", "file.mgf", "--mode", "invalid"], "invalid_value"),
    # ``--method`` is not parser-constrained (names are validated in the domain
    # layer), so the parser's honest complaint is the missing required ``-o``.
    (["similarity", "matrix", "file.mgf", "--method", "Unknown"], "missing_argument"),
]


@pytest.mark.parametrize("argv,expected_code", ACCEPTANCE_CASES)
def test_acceptance_invocation_produces_structured_error(argv, expected_code, capsys):
    exit_code, out, payload = run_cli(*argv, capsys=capsys)

    # stdout is exactly one JSON document (no non-JSON, no traceback)
    assert "Traceback (most recent call last)" not in out
    assert exit_code == 2, f"expected exit 2, got {exit_code} for {argv}"
    assert_envelope(payload, expected_code)


# -- structured parser fields ------------------------------------------------


def test_unknown_value_exposes_parameter_value_and_valid_values(capsys):
    _, _, payload = run_cli("similarity", "matrix", "file.mgf", "--mode", "invalid", capsys=capsys)
    error = payload["error"]
    assert error["code"] == "invalid_value"
    assert error["parameter"] == "--mode"
    assert error["value"] == "invalid"
    assert set(error["valid_values"]) == {"dense", "sparse"}


def test_unknown_command_exposes_valid_commands(capsys):
    _, _, payload = run_cli("unknown-command", capsys=capsys)
    error = payload["error"]
    assert error["code"] == "invalid_value"
    assert error["parameter"] == "<command>"
    assert error["value"] == "unknown-command"
    assert set(error["valid_values"]) == {"info", "filter", "similarity", "spectra"}


def test_missing_subcommand_exposes_valid_subcommands(capsys):
    _, _, payload = run_cli("similarity", capsys=capsys)
    error = payload["error"]
    assert error["code"] == "missing_subcommand"
    assert set(error["valid_values"]) == {
        "list", "info", "matrix", "build-index", "search",
    }


def test_unknown_flag_is_invalid_argument(capsys):
    _, _, payload = run_cli("similarity", "matrix", "file.mgf", "--unknown", capsys=capsys)
    error = payload["error"]
    assert error["code"] == "invalid_argument"
    assert error["parameter"] == "--unknown"


def test_missing_required_argument_names_the_parameter(capsys):
    _, _, payload = run_cli("similarity", "matrix", capsys=capsys)
    error = payload["error"]
    assert error["code"] == "missing_argument"
    assert "spectra_1" in error["parameter"]


# -- schema version on both success and error ---------------------------------


def test_error_payload_has_schema_version(capsys):
    _, _, payload = run_cli("filter", capsys=capsys)
    assert "schema_version" in payload


def test_success_payload_has_schema_version(capsys):
    exit_code, _, payload = run_cli("info", "--json", capsys=capsys)
    assert exit_code == 0
    assert payload["ok"] is True
    assert "schema_version" in payload


# -- domain (CliError) errors keep the envelope -------------------------------


def test_domain_error_uses_envelope_and_exit_1(capsys):
    # a real file-not-found domain error (not a parser error)
    exit_code, _, payload = run_cli("spectra", "describe", "/no/such/file.mgf", "--json", capsys=capsys)
    assert exit_code == EXIT_INTERNAL == 1
    assert_envelope(payload, "file_not_found")
    # domain errors carry structured detail
    assert payload["error"]["input_file"] == "/no/such/file.mgf"


# -- parser raises (never prints / sys.exit) ---------------------------------


def test_parser_raises_cli_error_instead_of_exiting(capsys):
    parser = build_parser()
    with pytest.raises(ArgparseError) as excinfo:
        parser.parse_args(["similarity", "matrix", "file.mgf", "--unknown"])
    assert excinfo.value.code == "invalid_argument"
    assert excinfo.value.exit_code == EXIT_USAGE


def test_parser_error_exit_code_is_usage(capsys):
    parser = build_parser()
    with pytest.raises(CliError) as excinfo:
        parser.parse_args(["unknown-command"])
    # an unknown command is a usage error with exit code 2
    assert excinfo.value.code == "invalid_value"
    assert excinfo.value.exit_code == EXIT_USAGE


def test_stderr_has_no_traceback_by_default(capsys):
    run_cli("similarity", "matrix", "file.mgf", "--unknown", capsys=capsys)
    err = capsys.readouterr().err
    assert "Traceback (most recent call last)" not in err


# -- operation is attached for parser errors deep in the tree -----------------


def test_parser_error_carries_operation(capsys):
    _, _, payload = run_cli("similarity", "matrix", "file.mgf", "--mode", "invalid", capsys=capsys)
    assert payload["error"]["operation"] == "similarity matrix"


# -- unexpected exceptions -> internal_error, no traceback --------------------


def test_unexpected_exception_is_internal_error(capsys, monkeypatch):
    import importlib

    cli_main = importlib.import_module("matchms.cli.main")

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    # main.py imports the command's run() directly, so patch the name it holds.
    monkeypatch.setattr(cli_main, "run_spectra_describe", boom)
    exit_code, out, payload = run_cli(
        "spectra", "describe", "/x.mgf", "--json", capsys=capsys
    )
    assert exit_code == EXIT_INTERNAL == 1
    assert payload["ok"] is False
    assert payload["error"]["code"] == "internal_error"
    # the traceback must not leak into the message or stdout
    assert "Traceback" not in out
    assert "kaboom" not in payload["error"]["message"]
