"""CLI command: matchms info (alias: inspect).

Reports matchms/Python versions, supported formats, available filters and
similarity methods, availability of indexed search and the CLI schema version.
"""

import platform
import matchms
from matchms import __version__ as matchms_version
from matchms.cli.introspection import filter_signature
from matchms.cli.output import CLI_JSON_SCHEMA_VERSION
from matchms.exporting.save_spectra import SUPPORTED_FILE_FORMATS as OUTPUT_FORMATS
from matchms.filtering.filter_order import ALL_FILTERS
from matchms.importing.load_spectra import SUPPORTED_FILE_FORMATS as INPUT_FORMATS


def _filter_inventory() -> list[dict]:
    items = []
    for order, func in enumerate(ALL_FILTERS):
        info = filter_signature(func)
        items.append(
            {
                "order": order,
                "name": func.__name__,
                "collection_supported": info["collection_supported"],
                "signature": info["signature"]["parameters"],
                "required_params": info["required"],
            }
        )
    return items


CLI_NAME = "matchms"

CLI_COMMANDS = {
    "info": "Report environment, supported formats and filters",
    "filter list": "List all available matchms filters with their default order",
    "filter pipelines": "List all available filter pipelines",
    "filter info": "Show the description and parameters of one filter",
    "filter run": "Run a filter pipeline on a spectra file (SpectraProcessor) and emit a report",
    "spectra convert": "Convert a spectra file between supported formats",
    "spectra describe": "Describe a spectra collection (peak counts, intensity sums, entropy)",
}


def run(args, ctx) -> int:
    payload = {
        "ok": True,
        "cli": {
            "name": CLI_NAME,
            "schema_version": CLI_JSON_SCHEMA_VERSION,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "commands": CLI_COMMANDS,
        },
        "matchms": {
            "version": matchms_version,
            "import_path": matchms.__file__,
        },
        "input_formats": INPUT_FORMATS,
        "output_formats": OUTPUT_FORMATS,
        "filters": _filter_inventory(),
    }

    if ctx.machine_mode:
        ctx.write_json(payload)
    else:
        ctx.write_text(_format_human(payload))
    return 0


def _format_human(payload: dict) -> str:
    lines = [
        f"matchms {payload['matchms']['version']} (Python {payload['cli']['python']})",
        "",
        "Input formats:  " + ", ".join(f".{k}" for k in sorted(payload["input_formats"])),
        "Output formats: " + ", ".join(f".{k}" for k in sorted(payload["output_formats"])),
        f"Filters:        {len(payload['filters'])} available "
        f"({sum(1 for f in payload['filters'] if f['collection_supported'])} collection-safe)",
    ]
    lines.append("")
    lines.append("Commands:")
    for cmd, desc in payload["cli"]["commands"].items():
        lines.append(f"  {cmd:<24} {desc}")
    lines.append("")
    lines.append("Run with --json for the machine-readable version.")
    return "\n".join(lines)
