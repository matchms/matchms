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
    with open(yaml_file, 'r', encoding="utf-8") as file:
        workflow = ordered_load(file, yaml.SafeLoader)
    if workflow["reference_filters"] == "processing_queries":
        workflow["reference_filters"] = workflow["query_filters"]
    return workflow
