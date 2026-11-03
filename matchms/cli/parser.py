"""A custom argparse parser whose errors are raised, not printed.

argparse's default behaviour — printing ``usage: ...`` / ``error: ...`` and
calling ``sys.exit(2)`` — would bypass the CLI's structured error boundary.
:class:`CliArgumentParser` replaces it: every parse failure is classified
(see :func:`matchms.cli.errors.classify_argparse_error`) and raised as an
:class:`~matchms.cli.errors.ArgparseError`, so parser failures flow through
the same renderer as runtime failures.

The parser also carries the context needed to structure parser errors that
argparse itself cannot express:

- ``matchms.parser.operation``: the "command subcommand" name being parsed
  (surfaced as ``error.operation`` in the error payload);
- ``matchms.parser.subcommands``: for a group parser, the valid subcommand
  names (surfaced as ``error.valid_values`` when a subcommand is missing).
"""

import argparse
import re
import sys
from matchms.cli.errors import ArgparseError, classify_argparse_error


#: Namespace key under which a parser stores its "command subcommand" name.
OPERATION_KEY = "matchms.parser.operation"
#: Namespace key under which a group parser stores its valid subcommand names.
SUBCOMMANDS_KEY = "matchms.parser.subcommands"

#: argparse's message for a missing required positional argument.
_RE_MISSING_SUBCOMMAND = re.compile(r"the following arguments are required: (<[A-Za-z_]+>)")
#: argparse's message for one or more missing required arguments.
_RE_REQUIRED_MESSAGE = re.compile(r"(?:the following arguments are required|missing required argument):")


class CliArgumentParser(argparse.ArgumentParser):
    """An :class:`argparse.ArgumentParser` that raises instead of exiting.

    Every parse failure is converted into an :class:`ArgparseError` (exit
    code 2); nothing is printed and the process is never terminated from
    here, so the error flows through the CLI's single structured renderer.
    Subparsers created via :meth:`add_parser` inherit this class, so the
    error boundary holds at every level of the command tree.
    """

    def error(self, message: str) -> ArgparseError:
        """Classify the parser error and raise it as an :class:`ArgparseError`.

        Replaces argparse's default behaviour of printing usage and calling
        ``sys.exit(2)``.

        argparse has a quirk: an unknown flag is only reported as
        "unrecognized arguments" if all required arguments were also
        supplied. Otherwise it first reports the missing required arguments,
        even though the unrecognized flag is what broke the parse. When the
        namespace therefore carries unrecognized option strings, that
        condition is reclassified as ``invalid_argument`` for the first
        unrecognized flag instead of ``missing_argument``.
        """
        subcommand_name = self._missing_subcommand_name(message)
        valid_values = self._subcommand_choices(subcommand_name)
        unrecognized = self._unrecognized_options()
        if unrecognized and _RE_REQUIRED_MESSAGE.search(message) and not subcommand_name:
            raise ArgparseError(
                message,
                "invalid_argument",
                operation=self._operation,
                parameter=unrecognized[0],
            )
        raise classify_argparse_error(
            message,
            operation=self._operation,
            valid_values=valid_values,
        )

    def parse_known_args(self, args=None, namespace=None):
        """Parse, remembering the args and namespace so :meth:`error` can use them.

        argparse keeps the in-flight unrecognized arguments only in local
        variables, so :meth:`error` re-derives them from the raw argument list
        it saw on the way in; the namespace is kept so a top-level failure can
        still recover the "command subcommand" that was being parsed.
        """
        active = list(args) if args is not None else sys.argv[1:]
        self._active_unrecognized = self._unrecognized_options_in(active)
        if namespace is None:
            namespace = argparse.Namespace()
        self._active_namespace = namespace
        return super().parse_known_args(active, namespace)

    # -- context attached by the parser builder (see set_parser_context) -----
    @property
    def _operation(self) -> str:
        """The "command subcommand" name of this parser ("" for top level)."""
        operation = self._defaults.get(OPERATION_KEY) or ""
        if operation:
            return operation
        # A failure reported by an ancestor (e.g. the top level, which only
        # knows the bare command) still has the leaf's operation in the
        # in-flight namespace: recover it so the error payload is specific.
        namespace = getattr(self, "_active_namespace", None)
        if namespace is not None:
            return getattr(namespace, OPERATION_KEY, "") or ""
        return ""

    def _subcommand_action(self, dest: str | None):
        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction) and (dest is None or action.dest == dest):
                return action
        return None

    def _missing_subcommand_name(self, message: str) -> str | None:
        """The metavar (e.g. ``<action>``) of a missing subcommand, if any."""
        match = _RE_MISSING_SUBCOMMAND.search(message)
        return match.group(1) if match else None

    def _unrecognized_options(self) -> list[str]:
        """Option strings this parser does not recognize, from the current parse."""
        return self._active_unrecognized or []

    def _unrecognized_options_in(self, arg_strings: list[str]) -> list[str]:
        """The option strings *this* parser does not recognize, mirroring
        argparse's own option/positional decision.

        For a parser that owns subcommands, everything after the first
        positional (the subcommand name) belongs to a subparser, so the scan
        stops there; otherwise the top level would misattribute subparser
        flags (e.g. ``--method``) to itself.
        """
        has_subparsers = any(isinstance(action, argparse._SubParsersAction) for action in self._actions)
        if has_subparsers:
            for arg_string in arg_strings:
                if arg_string and arg_string[0] not in self.prefix_chars:
                    break  # a positional starts the subcommand's argument list
                if self._is_unrecognized_option(arg_string):
                    return [arg_string]
            return []
        return [a for a in arg_strings if self._is_unrecognized_option(a)]

    def _is_unrecognized_option(self, arg_string: str) -> bool:
        """True if *arg_string* is an option string this parser does not know."""
        if not arg_string or arg_string[0] not in self.prefix_chars:
            return False
        if arg_string in self._option_string_actions:
            return False
        if "=" in arg_string and arg_string.split("=", 1)[0] in self._option_string_actions:
            return False
        if len(arg_string) == 1:
            return False  # single-dash singles are positionals to argparse
        if self._negative_number_matcher.match(arg_string) and not self._has_negative_number_optionals:
            return False
        return not self._get_option_tuples(arg_string)

    def _subcommand_choices(self, subcommand_name: str | None) -> list[str] | None:
        """The valid subcommand names of the group this parser was parsing."""
        if subcommand_name is None:
            return None
        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction) and action.metavar == subcommand_name:
                return sorted(action.choices)
        return None


def set_parser_context(
    parser: CliArgumentParser,
    *,
    operation: str = "",
    subcommands: list[str] | None = None,
) -> None:
    """Attach the CLI context (operation name, valid subcommands) to a parser.

    The context is stored both in the parser's defaults (so it ends up in the
    parsed namespace and is available in ``main`` after a *successful* parse)
    and in the parser's ``_defaults`` (so it is available when the parse
    *fails*).
    """
    parser.set_defaults(**{OPERATION_KEY: operation or None, SUBCOMMANDS_KEY: subcommands})
    parser._defaults[OPERATION_KEY] = operation or None
    if subcommands is not None:
        parser._defaults[SUBCOMMANDS_KEY] = subcommands
