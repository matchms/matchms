"""Unit tests for the shared human-readable output helpers."""

from matchms.cli.output import indent_text, parameters_table


def test_indent_text():
    assert indent_text("a\nb") == "  a\n  b"
    assert indent_text("a", prefix="") == "a"
    assert indent_text("") == ""


def test_parameters_table_renders_columns_and_cells():
    table = parameters_table(
        [
            {"name": "tolerance", "type": "float", "required": False, "default": 0.01, "description": "Max m/z diff."},
            {"name": "flag", "type": "bool", "required": True},
        ]
    )
    # header row with all columns
    for column in ("parameter", "type", "required", "default", "description"):
        assert column in table
    # first row with its values (default rendered as a float, bool as true/false)
    assert "tolerance" in table
    assert "float" in table
    assert "false" in table
    assert "0.01" in table
    assert "Max m/z diff." in table
    # missing optional keys fall back to empty cells without KeyError
    assert "flag" in table
    assert "true" in table


def test_parameters_table_empty():
    """No rows: only the header and separator lines are rendered."""
    table = parameters_table([])
    lines = table.splitlines()
    assert len(lines) == 2
    assert lines[0] == "parameter  type  required  default  description"
    assert set(lines[1]) == {"-", " "}
