from typing import Any, Dict

import torch
import torch.nn as nn
from pykan.kan.KAN import KAN as KANOriginal
from efficient_kan.src.efficient_kan.kan import KAN as KANEfficient
from chebykan.ChebyKANLayer import ChebyKANLayer

from .config import ModelType

T = torch.Tensor


class KANChebyshev(nn.Module):
    def __init__(self, in_features: int, out_features: int, degree: int) -> None:
        super().__init__()
        self.chebykan = ChebyKANLayer(in_features, out_features, degree=degree)

    def forward(self, x: T) -> T:
        batch_size, seq_length, _ = x.shape
        x = self.chebykan(x)
        return x.view(batch_size, seq_length, -1)


def get_model_cls(
    model_type: ModelType,
    use_kan_bias: bool,
    *,
    chebykan_degree: int = 4,
    **kwargs: Dict[str, Any],
) -> nn.Module:
    if model_type == ModelType.MLP:
        return lambda in_features, out_features, bias: nn.Linear(
            in_features, out_features, bias=bias
        )
    elif model_type == ModelType.KAN_ORIGINAL:
        return lambda in_features, out_features, bias: KANOriginal(
            width=[in_features, out_features], bias_trainable=use_kan_bias
        )
    elif model_type == ModelType.KAN_EFFICIENT:
        return lambda in_features, out_features, bias: KANEfficient(
            layers_hidden=[in_features, out_features]
        )
    elif model_type == ModelType.KAN_CHEBYSHEV:
        return lambda in_features, out_features, bias: KANChebyshev(
            in_features, out_features, degree=chebykan_degree
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
