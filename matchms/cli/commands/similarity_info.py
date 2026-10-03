"""CLI command: matchms similarity info.

Shows detailed information about a single similarity class from
``matchms.similarity``: which README group it belongs to, a short
description, its full docstring, the score fields it produces, the
computation methods it supports and its constructor parameters (name,
type, whether it is required, default value and description).
"""

from matchms.cli.commands.similarity_list import GROUPS, OTHER_GROUP_NAME
from matchms.cli.errors import raise_for_unknown_value
from matchms.cli.introspection import short_description, signature_parameters, similarity_signature
from matchms.cli.output import indent_text, parameters_table
from matchms.similarity import __all__ as SIMILARITY_NAMES
from matchms.similarity import get_similarity_function_by_name


CLI_COMMAND = "similarity info"


def _group_of(name: str) -> str:
    for group_name, _, class_names in GROUPS:
        if name in class_names:
            return group_name
    return OTHER_GROUP_NAME


def run(args, ctx) -> int:
    """Run the `similarity info` command."""
    name = args.similarity_name
    operation = f"{CLI_COMMAND} {name}"
    try:
        cls = get_similarity_function_by_name(name)
    except ValueError:
        raise_for_unknown_value(
            operation=operation,
            parameter="similarity_name",
            value=name,
            valid=sorted(SIMILARITY_NAMES),
            kind="similarity",
            hint="Use `matchms similarity list` to see all available similarity names.",
        )

    info = similarity_signature(cls)
    docstring = info["docstring"]
    parameters = signature_parameters(info)

    payload = {
        "ok": True,
        "operation": CLI_COMMAND,
        "name": name,
        "group": _group_of(name),
        "description": short_description(docstring),
        "score_fields": info["score_fields"],
        "is_commutative": info["is_commutative"],
        "methods": info["methods"],
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
        f"Similarity: {payload['name']}  (group: {payload['group']})",
        f"  {payload['description']}",
        f"  score fields: {', '.join(payload['score_fields'])}",
        f"  commutative: {'true' if payload['is_commutative'] else 'false'}",
        f"  methods: {', '.join(payload['methods'])}",
        "",
        "What it does:",
        indent_text(payload["docstring"] or "(no docstring)"),
        "",
        "Constructor parameters:",
    ]

    if payload["parameters"]:
        lines.append(parameters_table(payload["parameters"]))
    else:
        lines.append("  (no constructor parameters)")

    lines.append("")
    lines.append("Run with --json for the machine-readable version.")
    return "\n".join(lines)
