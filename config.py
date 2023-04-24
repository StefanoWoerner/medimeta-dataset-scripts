import os

from omegaconf import OmegaConf

default_conf = OmegaConf.load('default_config.yaml')
if os.path.exists('config.yaml'):
    yaml_conf = OmegaConf.load('config.yaml')
else:
    yaml_conf = OmegaConf.create()
cli_conf = OmegaConf.from_cli()

config = OmegaConf.merge(default_conf, yaml_conf, cli_conf)
