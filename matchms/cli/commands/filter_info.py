"""CLI command: matchms filter info.

Shows detailed information about a single matchms filter: where it sits in the
default execution order, a short description, its full docstring and the
user-settable parameters (name, type, whether it is required, default value and
per-parameter documentation).
"""

from matchms.cli.errors import raise_for_unknown_value
from matchms.cli.introspection import filter_signature, short_description, signature_parameters
from matchms.cli.output import indent_text, parameters_table
from matchms.filtering.filter_order import ALL_FILTERS, FILTER_FUNCTION_NAMES


CLI_COMMAND = "filter info"


def run(args, ctx) -> int:
    """Run the `filter info` command."""
    name = args.filter_name
    operation = f"{CLI_COMMAND} {name}"
    if name not in FILTER_FUNCTION_NAMES:
        raise_for_unknown_value(
            operation=operation,
            parameter="filter_name",
            value=name,
            valid=FILTER_FUNCTION_NAMES,
            kind="filter",
            hint="Use `matchms filter list` to see all available filter names.",
        )

    func = FILTER_FUNCTION_NAMES[name]
    info = filter_signature(func)
    docstring = info["docstring"]
    parameters = signature_parameters(info, skip=("clone",))

    payload = {
        "ok": True,
        "operation": CLI_COMMAND,
        "name": name,
        "order": ALL_FILTERS.index(func),
        "description": short_description(docstring),
        "collection_supported": info["collection_supported"],
        "docstring": docstring,
        "required": [p["name"] for p in parameters if p["required"]],
        "parameters": parameters,
    }

    if ctx.machine_mode:
        ctx.write_json(payload)
    else:
        ctx.write_text(_format_human(payload))
    return 0


def _format_human(payload: dict) -> str:
    lines = [
        f"Filter: {payload['name']}  (position {payload['order']} in the default filter order)",
        f"  {payload['description']}",
        f"  collection-safe: {'true' if payload['collection_supported'] else 'false'}",
        "",
        "What it does:",
        indent_text(payload["docstring"] or "(no docstring)"),
        "",
        "Parameters:",
    ]

    if payload["parameters"]:
        lines.append(parameters_table(payload["parameters"]))
    else:
        lines.append("  (no user-settable parameters)")

    lines.append("")
    lines.append("Run with --json for the machine-readable version.")
    return "\n".join(lines)
