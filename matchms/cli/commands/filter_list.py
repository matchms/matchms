"""CLI command: matchms filter list.

Lists all available matchms filters in their default execution order
(:data:`~matchms.filtering.filter_order.ALL_FILTERS`), showing for each filter
its order, name and a short description.
"""

from matchms.cli.introspection import _clean_docstring
from matchms.cli.output import human_table
from matchms.filtering.filter_order import ALL_FILTERS


CLI_COMMAND = "filter list"


def _short_description(docstring: str) -> str:
    """Return the leading sentence of a filter's docstring."""
    first_line = docstring.split("\n", 1)[0].strip() if docstring else ""
    if first_line:
        return first_line
    # Fall back to the first sentence of the docstring body.
    text = " ".join(docstring.split()) if docstring else ""
    for end in (".", "!", "?"):
        idx = text.find(end)
        if idx != -1:
            return text[: idx + 1]
    return text


def _filter_rows() -> list[dict]:
    rows = []
    for order, func in enumerate(ALL_FILTERS):
        rows.append(
            {
                "order": order,
                "name": func.__name__,
                "description": _short_description(_clean_docstring(func.__doc__)),
            }
        )
    return rows


def run(args, ctx) -> int:
    """Run the `filter list` command."""
    rows = _filter_rows()
    payload = {
        "ok": True,
        "operation": CLI_COMMAND,
        "n_filters": len(rows),
        "filters": rows,
    }

    if ctx.machine_mode:
        ctx.write_json(payload)
    else:
        ctx.write_text(_format_human(rows))
    return 0


def _format_human(rows: list[dict]) -> str:
    table = human_table(
        ["order", "name", "description"],
        [[row["order"], row["name"], row["description"]] for row in rows],
    )
    return "\n".join(
        [
            f"Available filters ({len(rows)}), in default execution order:",
            table,
            "",
            "Run with --json for the machine-readable version.",
        ]
    )
