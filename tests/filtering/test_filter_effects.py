import ast
import os
from matchms.filtering.filter_effects import FILTER_EFFECTS


def test_filter_effects_is_complete():
    """Check that FILTER_EFFECTS contains all available matchms filters."""

    def get_functions_from_file(file_path):
        """Return names of all top-level functions in a Python file."""
        with open(file_path, encoding="utf-8") as file:
            tree = ast.parse(file.read(), filename=file_path)

        return [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        ]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    filtering_directory = os.path.join(current_dir, "../../matchms/filtering")
    directories_with_filters = ["metadata_processing", "peak_processing"]

    filter_function_names = []

    for directory_name in directories_with_filters:
        directory = os.path.join(filtering_directory, directory_name)

        for script in os.listdir(directory):
            if script.startswith("_"):
                continue
            if not script.endswith(".py"):
                continue

            functions = get_functions_from_file(os.path.join(directory, script))

            for function in functions:
                if not function.startswith("_"):
                    filter_function_names.append((script, function))

    # default_filters.py lives directly in matchms/filtering.
    default_filters_file = os.path.join(filtering_directory, "default_filters.py")
    for function in get_functions_from_file(default_filters_file):
        if not function.startswith("_"):
            filter_function_names.append(("default_filters.py", function))

    for script, filter_function in filter_function_names:
        assert filter_function in FILTER_EFFECTS, (
            f"The filter {filter_function} in {script} is not included in "
            "FILTER_EFFECTS. Add its filter effect to filter_effects.py. "
            "If this function is not a public filter, prefix its name with "
            "an underscore."
        )


def test_filter_effects_are_valid():
    valid_effects = {"metadata", "fragments", "remove"}

    for filter_name, effect in FILTER_EFFECTS.items():
        assert effect in valid_effects, (
            f"Invalid effect {effect!r} for filter {filter_name!r}. "
            f"Expected one of {valid_effects}."
        )