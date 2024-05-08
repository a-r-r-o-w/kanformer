import torch.nn as nn
from pykan.kan.KAN import KAN as KANOriginal
from efficient_kan.src.efficient_kan.kan import KAN as KANEfficient
from .config import ModelType


def get_model_cls(model_type: ModelType, use_kan_bias: bool) -> nn.Module:
    if model_type == ModelType.MLP:
        return lambda in_features, out_features, bias: nn.Linear(
            in_features, out_features, bias=bias
        )
    elif model_type == ModelType.KAN_ORIGINAL:
        return lambda in_features, out_features, bias: KANOriginal(
            width=[in_features, out_features], bias_trainable=use_kan_bias
        )
    else:
        return lambda in_features, out_features, bias: KANEfficient(
            layers_hidden=[in_features, out_features]
        )
