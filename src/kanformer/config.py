from enum import Enum


class ModelType(str, Enum):
    MLP = "mlp"
    KAN_ORIGINAL = "kan_original"
    KAN_EFFICIENT = "kan_efficient"
