from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from pykan.kan.KAN import KAN as KANOriginal
from efficient_kan.src.efficient_kan.kan import KAN as KANEfficient
from chebykan.ChebyKANLayer import ChebyKANLayer
from fastkan.fastkan import FastKANLayer

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


class KANFast(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.fastkan = FastKANLayer(
            input_dim=in_features,
            output_dim=out_features,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_base_update=use_base_update,
            base_activation=base_activation,
            spline_weight_init_scale=spline_weight_init_scale,
        )

    def forward(self, x: T, time_benchmark: bool = False) -> T:
        return self.fastkan(x, time_benchmark=time_benchmark)


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
    elif model_type == ModelType.KAN_FAST:
        return lambda in_features, out_features, bias: KANFast(
            in_features, out_features
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
