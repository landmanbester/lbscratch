from dataclasses import dataclass
import glob
import os.path

from omegaconf.omegaconf import OmegaConf
from scabha import configuratt
from scabha.cargo import Parameter

schema = None


@dataclass
class _CabInputsOutputs:
    inputs: dict[str, Parameter]
    outputs: dict[str, Parameter]


# load schema files
if schema is None:
    # all *.yaml files under lbscratch.parser will be loaded automatically

    files = glob.glob(os.path.join(os.path.dirname(__file__), "*.yaml"))

    structured = OmegaConf.structured(_CabInputsOutputs)

    tmp = configuratt.load_nested(
        files, structured=structured, config_class="PfbCleanCabs", use_cache=False
    )

    # this is required since upgrade of scabha to caching branch
    # tmp is a tuple containing the config object as the first element
    # and a set containing locations of .yaml configs for pfb workers
    schema = OmegaConf.create(tmp[0])

    for worker in schema.keys():
        for param in schema[worker]["inputs"]:
            if schema[worker]["inputs"][param]["default"] == "<UNSET DEFAULT VALUE>":
                schema[worker]["inputs"][param]["default"] = None
