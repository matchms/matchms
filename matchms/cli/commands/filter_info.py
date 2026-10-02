"""CLI command: matchms filter info.

Shows detailed information about a single matchms filter: where it sits in the
default execution order, a short description, its full docstring and the
user-settable parameters (name, type, whether it is required, default value and
per-parameter documentation).
"""

from matchms.cli.errors import raise_for_unknown_value
from matchms.cli.introspection import filter_signature
from matchms.cli.output import human_table
from matchms.filtering.filter_order import ALL_FILTERS, FILTER_FUNCTION_NAMES


CLI_COMMAND = "filter info"


def _short_description(docstring: str) -> str:
    first_line = docstring.split("\n", 1)[0].strip() if docstring else ""
    if first_line:
        return first_line
    text = " ".join(docstring.split()) if docstring else ""
    for end in (".", "!", "?"):
        idx = text.find(end)
        if idx != -1:
            return text[: idx + 1]
    return text


def _parameters(info: dict) -> list[dict]:
    param_docs = info["param_docs"]
    parameters = []
    for name, spec in info["signature"]["parameters"].items():
        # ``clone`` is managed by the SpectraProcessor, not by the user.
        if name == "clone":
            continue
        entry = {"name": name, "type": spec["type"], "required": spec["required"]}
        if "default" in spec:
            entry["default"] = spec["default"]
        if name in param_docs:
            entry["description"] = param_docs[name]
        parameters.append(entry)
    return parameters


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
    parameters = _parameters(info)

    payload = {
        "ok": True,
        "operation": CLI_COMMAND,
        "name": name,
        "order": ALL_FILTERS.index(func),
        "description": _short_description(docstring),
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
        _indent(payload["docstring"] or "(no docstring)"),
        "",
        "Parameters:",
    ]

    if payload["parameters"]:
        lines.append(
            human_table(
                ["parameter", "type", "required", "default", "description"],
                [
                    [
                        p["name"],
                        p["type"],
                        p["required"],
                        p.get("default"),
                        p.get("description", ""),
                    ]
                    for p in payload["parameters"]
                ],
            )
        )
    else:
        lines.append("  (no user-settable parameters)")

    lines.append("")
    lines.append("Run with --json for the machine-readable version.")
    return "\n".join(lines)


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())
