import argparse
from matchms import __version__ as matchms_version
from matchms.cli.commands.info import run as run_info
from matchms.cli.commands.spectra_convert import EXPORT_STYLES
from matchms.cli.commands.spectra_convert import run as run_spectra_convert
from matchms.cli.commands.spectra_describe import run as run_spectra_describe
from matchms.cli.errors import CliError, emit_error, unexpected_error_payload
from matchms.cli.output import CLI_JSON_SCHEMA_VERSION, OutputContext


def _add_common_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Add --json/--table/--quiet/--verbose to a parser.

    When *suppress* is True (subparsers), defaults are SUPPRESS so that flags
    given at the top level are not clobbered by subparser defaults.
    """
    default = argparse.SUPPRESS if suppress else None

    group = parser.add_argument_group(
        "output modes",
        "Machine mode (--json) prints stable JSON on stdout. Human mode (--table) "
        "prints readable text. Without either flag, JSON is used when stdout is "
        "not a terminal.",
    )
    group.add_argument(
        "--json",
        action="store_true",
        default=default,
        help="Force machine-readable JSON output on stdout.",
    )
    group.add_argument(
        "--table",
        action="store_true",
        default=default,
        help="Force human-readable tabular output on stdout.",
    )
    group.add_argument(
        "--quiet",
        action="store_true",
        default=False if not suppress else argparse.SUPPRESS,
        help="Suppress log output (only errors are reported).",
    )
    group.add_argument(
        "--verbose",
        action="store_true",
        default=False if not suppress else argparse.SUPPRESS,
        help="Enable debug-level logging (goes to stderr).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="matchms",
        description=(
            "matchms command line interface"
        ),
        epilog=(
            "Examples:\n"
            "  matchms info --json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"matchms {matchms_version} (CLI JSON schema v{CLI_JSON_SCHEMA_VERSION})",
        help="Print the matchms version and the CLI JSON schema version.",
    )
    _add_common_flags(parser, suppress=False)

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # common flags that also work after the subcommand
    common = argparse.ArgumentParser(add_help=False)
    _add_common_flags(common, suppress=True)

    # -- info  --------------------------------------------------------------
    p = subparsers.add_parser(
        "info",
        parents=[common],
        help="Report versions, supported formats and filters",
        description=(
            "Report the matchms/Python versions, supported input/output formats, "
            "all filters (with default order), all similarity methods, availability "
            "of indexed search and the CLI JSON schema version."
        ),
    )
    p.set_defaults(func=run_info)

    # -- spectra  ----------------------------------------------------------
    p = subparsers.add_parser(
        "spectra",
        parents=[common],
        help="Inspect and convert spectra files",
        description=(
            "Work with spectra files. `spectra describe` reports descriptive "
            "statistics of the loaded collection; `spectra convert` converts a "
            "spectra file to another supported format."
        ),
    )
    spectra_subparsers = p.add_subparsers(dest="spectra_command", metavar="<action>")

    # spectra describe -----------------------------------------------------
    pd = spectra_subparsers.add_parser(
        "describe",
        parents=[common],
        help="Describe a spectra collection (peak counts, intensity sums, entropy)",
        description=(
            "Load the spectra file with load_ms2_dataset and report the "
            "descriptive statistics of the collection (peak counts, intensity "
            "sums, intensity entropy) as computed by SpectraCollection.describe()."
        ),
    )
    pd.add_argument(
        "spectrumfile",
        help="Path to the spectra file to describe "
        "(supported extensions: json, mgf, msp, mzml, mzxml, pickle).",
    )
    pd.add_argument(
        "--ftype",
        default="auto",
        help="File type to use for import. Defaults to 'auto', which guesses the "
        "file type from the file extension.",
    )
    pd.set_defaults(func=run_spectra_describe)

    # spectra convert ------------------------------------------------------
    pc = spectra_subparsers.add_parser(
        "convert",
        parents=[common],
        help="Convert a spectra file to another supported format",
        description=(
            "Load the input spectra file with load_ms2_dataset and save it with "
            "save_spectra in the format of the output file extension "
            "(e.g. in.mgf -> out.msp). The output file must not exist unless "
            "--append is passed (append is only supported for .mgf and .msp)."
        ),
    )
    pc.add_argument(
        "input",
        help="Path to the input spectra file "
        "(supported extensions: json, mgf, msp, mzml, mzxml, pickle).",
    )
    pc.add_argument(
        "output",
        help="Path to the output spectra file "
        "(supported extensions: json, mgf, msp, pickle).",
    )
    pc.add_argument(
        "--ftype",
        default="auto",
        help="File type to use for import. Defaults to 'auto', which guesses the "
        "file type from the input file extension.",
    )
    pc.add_argument(
        "--export-style",
        default="matchms",
        choices=EXPORT_STYLES,
        help="Metadata key style used for the export (default: %(default)s).",
    )
    pc.add_argument(
        "--append",
        action="store_true",
        help="Append to the output file instead of overwriting it. Only supported "
        "for .mgf and .msp output files.",
    )
    pc.set_defaults(func=run_spectra_convert)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        parser.print_help()
        return 2

    # configure_logging(_log_level(args))
    ctx = OutputContext(
        json_flag=getattr(args, "json", None),
        table_flag=getattr(args, "table", None),
        quiet=bool(getattr(args, "quiet", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )

    try:
        return args.func(args, ctx)
    except CliError as exc:
        emit_error(ctx, exc.to_dict())
        return exc.exit_code
    except Exception as exc:
        operation = getattr(args, "command", "") or ""
        emit_error(ctx, unexpected_error_payload(exc, operation))
        return 1
