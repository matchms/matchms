"""Output-mode selection on parse errors and exit-safety of ``main``.

These tests lock in the requirements for honouring ``--json``/``--table`` when
the parse fails and for making :func:`matchms.cli.main.main` exit-safe:

* R1/R3 — a parse failure honours the output flags recovered from the raw argv
  (position-independent; ``--json`` wins over ``--table``);
* R2   — when no flag is given the mode falls back to terminal auto-detection;
* R4   — the missing-command error uses the parsed args' flags;
* R5   — ``--help``/``--version`` return an int exit status and never raise
  ``SystemExit``.

The machine-vs-human decision normally depends on ``sys.stdout.isatty()``. An
agent harness that runs the CLI in a PTY sees ``isatty() == True`` (a terminal)
and would therefore get plain text even though it passed ``--json``. These
tests force ``isatty() == True`` to prove the explicit flags win over that
auto-detection.
"""

import json
import pytest
from matchms.cli.main import _output_flags, main


class _TTY:
    """A writable stream that pretends to be a terminal (``isatty() is True``)."""

    def __init__(self) -> None:
        self._buffer = []

    def write(self, text: str) -> int:
        self._buffer.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True

    def getvalue(self) -> str:
        return "".join(self._buffer)


def run_in_tty(argv, monkeypatch):
    """Run ``main(argv)`` with a TTY stdout, returning ``(exit_code, stdout)``.

    ``sys.stdout`` is swapped for a :class:`_TTY` (so ``isatty()`` is True) for
    the duration of the call and restored afterwards. stderr is left as-is (the
    concise error line is not asserted here).
    """
    tty = _TTY()
    monkeypatch.setattr("sys.stdout", tty)
    exit_code = main(list(argv))
    return exit_code, tty.getvalue()


# -- R1/R3: parse errors honour --json on a TTY ------------------------------
# Each entry: (argv, expected error code). On a TTY (isatty()==True) the only
# reason a JSON error is emitted is that --json was honoured on the failed
# parse; every one of these writes exactly one JSON document to stdout.
JSON_ON_TTY_CASES = [
    (["similarity", "matrix", "file.mgf", "--json"], "missing_argument"),
    (["similarity", "matrix", "file.mgf", "--mode", "invalid", "--json"], "invalid_value"),
    (["similarity", "matrix", "file.mgf", "--unknown", "--json"], "invalid_argument"),
    (["--json", "unknown-command"], "invalid_value"),
]


@pytest.mark.parametrize("argv,expected_code", JSON_ON_TTY_CASES)
def test_parse_error_honours_json_on_tty(argv, expected_code, monkeypatch):
    exit_code, out = run_in_tty(argv, monkeypatch)

    # exactly one JSON document (no leading/trailing non-JSON, no traceback)
    assert "Traceback (most recent call last)" not in out
    payload = json.loads(out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == expected_code
    assert exit_code == 2


def test_json_before_and_after_subcommand_behave_the_same(monkeypatch):
    before, _ = run_in_tty(["--json", "similarity", "matrix", "file.mgf"], monkeypatch)
    after, _ = run_in_tty(["similarity", "matrix", "file.mgf", "--json"], monkeypatch)
    assert before == after == 2


# -- R4: a bare `matchms --json` reports missing_command as JSON on a TTY ----


def test_bare_matchms_json_is_missing_command_on_tty(monkeypatch):
    exit_code, out = run_in_tty(["--json"], monkeypatch)
    payload = json.loads(out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "missing_command"
    assert exit_code == 2


# -- R2: without a flag the mode falls back to terminal auto-detection --------


def test_no_flag_on_tty_falls_back_to_human_mode(monkeypatch):
    # isatty() is True and no --json/--table was given -> auto-detect -> human
    exit_code, out = run_in_tty(["similarity", "matrix"], monkeypatch)
    assert exit_code == 2
    # human mode writes plain text, not JSON
    assert not out.lstrip().startswith("{")
    assert "error:" in out


# -- R3: --json wins over --table --------------------------------------------


def test_json_wins_over_table_on_tty(monkeypatch):
    exit_code, out = run_in_tty(["--json", "--table", "similarity", "matrix"], monkeypatch)
    payload = json.loads(out)
    assert payload["ok"] is False
    assert exit_code == 2


# -- R5: --help/--version return an int and never raise SystemExit ------------


@pytest.mark.parametrize("argv", [["--help"], ["--version"]])
def test_help_and_version_return_int_and_print_plain_text(argv, monkeypatch):
    exit_code, out = run_in_tty(argv, monkeypatch)
    assert exit_code == 0
    assert isinstance(exit_code, int)
    # plain text, not a JSON envelope
    assert not out.lstrip().startswith("{")
    assert out.strip()


def test_main_never_raises_system_exit(monkeypatch):
    for argv in (["--help"], ["--version"], [], ["no-such-command"]):
        # would raise SystemExit if uncaught; must return an int instead
        assert isinstance(run_in_tty(argv, monkeypatch)[0], int)


# -- unit: _output_flags is position-independent and prefers JSON --------------


def test_output_flags_are_position_independent():
    assert _output_flags(["--json", "cmd"]) == (True, None, False, False)
    assert _output_flags(["cmd", "--json"]) == (True, None, False, False)
    assert _output_flags(["--table", "cmd", "--verbose"]) == (None, True, False, True)
    assert _output_flags(["cmd"]) == (None, None, False, False)


def test_output_flags_reports_both_json_and_table():
    # both are reported; OutputContext.machine_mode resolves the precedence
    json_flag, table_flag, _, _ = _output_flags(["--json", "--table", "cmd"])
    assert json_flag is True
    assert table_flag is True
