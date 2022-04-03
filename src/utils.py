from detectron2.config import CfgNode


def load_config(filepath: str):
    config_dict = CfgNode.load_yaml_with_base(filename=filepath)
    return CfgNode(init_dict=config_dict)
