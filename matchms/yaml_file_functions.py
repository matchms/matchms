from collections import OrderedDict
import yaml


def ordered_load(stream, loader=yaml.SafeLoader, object_pairs_hook=OrderedDict) -> OrderedDict:
    """ Code from https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    """
    class OrderedLoader(loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def ordered_dump(data: OrderedDict, stream=None, dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def load_workflow_from_yaml_file(yaml_file: str) -> OrderedDict:
    """Load a Pipeline workflow from YAML.

    Expected keys are:
    - spectra_1_filters
    - spectra_2_filters
    - score_computations

    For convenience, spectra_2_filters may be set to the string
    "processing_spectra_1" to reuse spectra_1_filters.
    """
    with open(yaml_file, "r", encoding="utf-8") as file:
        workflow = ordered_load(file, yaml.SafeLoader)

    if workflow is None:
        raise ValueError(f"Workflow file {yaml_file} is empty.")

    if not isinstance(workflow, OrderedDict):
        workflow = OrderedDict(workflow)

    expected_keys = {"spectra_1_filters", "spectra_2_filters", "score_computations"}
    if set(workflow.keys()) != expected_keys:
        raise ValueError(
            f"Workflow must contain exactly keys {expected_keys}, "
            f"but got {set(workflow.keys())}."
        )

    if workflow["spectra_2_filters"] == "processing_spectra_1":
        workflow["spectra_2_filters"] = workflow["spectra_1_filters"]

    return workflow
