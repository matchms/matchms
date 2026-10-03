"""CLI command: matchms similarity list.

Lists all similarity classes exposed by ``matchms.similarity``, grouped like
the "Similarity measures" section of the README (Cosine, Modified cosine,
Spectral entropy, ...). Each row shows the group, the class name, a short
description and the supported computation methods.

Classes that the README does not group are listed under the ``Other`` group;
no information about them is invented beyond their own docstring.
"""

from matchms.cli.introspection import similarity_signature
from matchms.cli.output import human_table
from matchms.similarity import __all__ as SIMILARITY_NAMES
from matchms.similarity import get_similarity_function_by_name


CLI_COMMAND = "similarity list"

OTHER_GROUP_NAME = "Other"

# Groups as documented in the README ("Similarity measures" section):
# (group name, "typical use" description, ordered member classes).
GROUPS = (
    (
        "Cosine",
        "Standard peak-based spectral similarity with one-to-one peak matching.",
        ["Cosine", "CosineGreedy", "CosineHungarian", "CosineLinear", "CosineFlash", "CosineBlink"],
    ),
    (
        "Modified cosine",
        "Cosine similarity allowing both direct fragment matches and matches "
        "shifted by the difference in precursor m/z.",
        ["ModifiedCosine", "ModifiedCosineGreedy", "ModifiedCosineHungarian", "ModifiedCosineLinear"],
    ),
    (
        "Spectral entropy",
        "General-purpose spectral entropy similarity with explicit one-to-one matching.",
        ["Entropy", "EntropyGreedy", "EntropyFlash"],
    ),
    (
        "Search-optimized spectral entropy",
        "High-throughput fragment-only entropy searches against large reference libraries.",
        ["EntropySearch"],
    ),
    (
        "Neutral-loss cosine",
        "Compare spectra based on neutral-loss rather than fragment m/z patterns.",
        ["NeutralLossesCosine"],
    ),
    (
        "Binned spectra",
        "Compare fixed-width binned spectrum representations using cosine or Euclidean similarity.",
        ["BinnedEmbeddingSimilarity"],
    ),
    (
        "Molecular structure",
        "Compare molecular fingerprints derived from structure metadata.",
        ["FingerprintSimilarity"],
    ),
    (
        "Metadata",
        "Compare arbitrary metadata fields using exact or tolerance-based matching.",
        ["MetadataMatch"],
    ),
    (
        "Precursor or parent mass",
        "Simple matching based on precursor m/z or parent mass.",
        ["PrecursorMzMatch", "ParentMassMatch"],
    ),
)


def _short_description(docstring: str) -> str:
    """Return the leading sentence of a docstring."""
    first_line = docstring.split("\n", 1)[0].strip() if docstring else ""
    if first_line:
        return first_line
    text = " ".join(docstring.split()) if docstring else ""
    for end in (".", "!", "?"):
        idx = text.find(end)
        if idx != -1:
            return text[: idx + 1]
    return text


def _row(name: str, group: str) -> dict:
    info = similarity_signature(get_similarity_function_by_name(name))
    return {
        "group": group,
        "name": name,
        "description": _short_description(info["docstring"]),
        "score_fields": info["score_fields"],
        "methods": info["methods"],
    }


def _rows() -> list[dict]:
    rows = []
    grouped = set()
    for group_name, _, class_names in GROUPS:
        for name in class_names:
            grouped.add(name)
            rows.append(_row(name, group_name))
    # Classes that exist in the code but in none of the README groups.
    for name in SIMILARITY_NAMES:
        if name not in grouped:
            rows.append(_row(name, OTHER_GROUP_NAME))
    return rows


def run(args, ctx) -> int:
    """Run the `similarity list` command."""
    rows = _rows()
    payload = {
        "ok": True,
        "operation": CLI_COMMAND,
        "n_similarities": len(rows),
        "similarities": rows,
    }

    if ctx.machine_mode:
        ctx.write_json(payload)
    else:
        ctx.write_text(_format_human(rows))
    return 0


def _format_human(rows: list[dict]) -> str:
    table_rows = [
        [row["group"], row["name"], row["description"], row["methods"]]
        for row in rows
    ]
    return "\n".join(
        [
            f"Available similarity measures ({len(rows)}):",
            "",
            human_table(["group", "name", "description", "methods"], table_rows),
            "",
            "Show the details of one similarity with `matchms similarity info <name>`.",
            "",
            "Run with --json for the machine-readable version.",
        ]
    )
