from copy import deepcopy
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from .configuration_tsrt import TSRTConfig

def tsrt_config_to_qwen3_config(
    config: TSRTConfig,
) -> Qwen3Config:
    """
    Convert TSRTConfig -> Qwen3Config
    bằng cách copy toàn bộ field chung.
    """

    qwen_fields = {
        k
        for k in Qwen3Config.__annotations__.keys()
    }

    kwargs = {}

    for field in qwen_fields:
        if hasattr(config, field):
            kwargs[field] = deepcopy(
                getattr(config, field)
            )

    return Qwen3Config(**kwargs)