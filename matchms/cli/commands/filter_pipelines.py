"""CLI command: matchms filter pipelines.

Lists all available pipeline constants as defined in
:mod:`matchms.filtering.default_pipelines` (and therefore selectable with
``--pipeline`` of ``matchms filter run``), showing for each pipeline its
name, a short description and the number of filters it contains.

Pass a pipeline name to show the filters of that pipeline in execution
order (including any pipeline-specific parameters).
"""

from matchms.cli.errors import raise_for_unknown_value
from matchms.cli.output import human_table
from matchms.filtering.default_pipelines import FILTER_SETS_BY_NAME


CLI_COMMAND = "filter pipelines"

PIPELINES = FILTER_SETS_BY_NAME

# Pipeline that `matchms filter run` uses when --pipeline is omitted.
DEFAULT_PIPELINE = "DEFAULT_FILTERS"

_DESCRIPTIONS = {
    "HARMONIZE_METADATA_FIELD_NAMES": (
        "Harmonize the metadata field names (charge, compound name, pepmass, precursor mz, retention time/index)."
    ),
    "DERIVE_METADATA_IN_WRONG_FIELD": (
        "Derive adduct, formula, ion mode and InChI entries from the compound name and clean it."
    ),
    "HARMONIZE_METADATA_ENTRIES": (
        "Harmonize missing annotation entries (inchi/inchikey/smiles) and clean the adduct."
    ),
    "DERIVE_MISSING_METADATA": (
        "Derive missing metadata (charge, parent mass, InChI, formula) from what is already present."
    ),
    "REQUIRE_COMPLETE_METADATA": "Require a precursor mz and a valid ion mode, removing spectra without them.",
    "REPAIR_ANNOTATION": (
        "Repair the annotation (smiles of salts, parent mass, adduct) and derive annotation from the compound name."
    ),
    "REQUIRE_COMPLETE_ANNOTATION": (
        "Require the parent mass matching the smiles, a valid annotation and a consistent adduct/ion mode."
    ),
    "CLEAN_PEAKS": "Clean the peaks (relative intensity, remove noise, reduce to the most intense peaks).",
    "OTHER_FILTERS": "Filter sets that are part of none of the pipelines above.",
    "BASIC_FILTERS": (
        "Basic metadata harmonization and derivation (field names, wrong-field derivation, entry harmonization)."
    ),
    "DEFAULT_FILTERS": "Basic filters plus intensity normalization, complete metadata and derived missing metadata.",
    "LIBRARY_CLEANING": "Default filters plus annotation repair, complete annotation and a correct ms level.",
    "MS2DEEPSCORE_TRAINING": "Library cleaning plus peak cleaning, as used for ms2deepscore training.",
}


def _step_label(step: dict) -> str:
    """Render one filter step, appending its parameters when present."""
    if step.get("parameters"):
        params = ", ".join(f"{k}={v}" for k, v in step["parameters"].items())
        return f"{step['name']} ({params})"
    return step["name"]


def _pipeline_steps(filter_set: list) -> list[dict]:
    """Build the JSON-safe filter steps of one pipeline, in execution order."""
    steps = []
    for entry in filter_set:
        if isinstance(entry, (tuple, list)):
            steps.append({"name": entry[0].__name__, "parameters": dict(entry[1])})
        else:
            steps.append({"name": entry.__name__})
    return steps


def _overview_rows() -> list[dict]:
    """Build the JSON-safe overview row for each pipeline (without filter steps)."""
    rows = []
    for name, filter_set in PIPELINES.items():
        rows.append(
            {
                "name": name,
                "description": _DESCRIPTIONS.get(name, ""),
                "n_filters": len(filter_set),
                "is_default": name == DEFAULT_PIPELINE,
            }
        )
    return rows


def _detail_payload(name: str) -> dict:
    """Build the JSON-safe detail of one pipeline, including its filter steps."""
    steps = _pipeline_steps(PIPELINES[name])
    return {
        "name": name,
        "description": _DESCRIPTIONS.get(name, ""),
        "n_filters": len(steps),
        "is_default": name == DEFAULT_PIPELINE,
        "filters": steps,
    }


def run(args, ctx) -> int:
    """Run the `filter pipelines` command."""
    name = args.pipeline_name
    if name is None:
        rows = _overview_rows()
        payload = {
            "ok": True,
            "operation": CLI_COMMAND,
            "default_pipeline": DEFAULT_PIPELINE,
            "n_pipelines": len(rows),
            "pipelines": rows,
        }
    else:
        operation = f"{CLI_COMMAND} {name}"
        if name not in PIPELINES:
            raise_for_unknown_value(
                operation=operation,
                parameter="pipeline_name",
                value=name,
                valid=PIPELINES,
                kind="pipeline",
                hint="Use `matchms filter pipelines` without arguments to see all available pipeline names.",
            )
        payload = {"ok": True, "operation": operation, **_detail_payload(name)}

    if ctx.machine_mode:
        ctx.write_json(payload)
    else:
        ctx.write_text(_format_human(payload))
    return 0


def _format_human(payload: dict) -> str:
    if "filters" in payload:
        lines = [
            f"Pipeline: {payload['name']}  ({payload['n_filters']} filters)",
            f"  {payload['description']}",
            "",
            "Filters (in execution order):",
            f"  {payload['name']}:",
        ]
        lines.extend(f"    - {_step_label(step)}" for step in payload["filters"])
        lines += [
            "",
            "Run with --json for the machine-readable version.",
        ]
        return "\n".join(lines)

    lines = [
        f"Available pipelines ({len(payload['pipelines'])}), "
        f"default for `filter run` is {payload['default_pipeline']}:",
        "",
        human_table(
            ["name", "n_filters", "description"],
            [[row["name"], row["n_filters"], row["description"]] for row in payload["pipelines"]],
        ),
        "",
        "Show the filters of one pipeline with `matchms filter pipelines <name>`.",
        "",
        "Run with --json for the machine-readable version.",
    ]
    return "\n".join(lines)
