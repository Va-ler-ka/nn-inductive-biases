"""Полносвязная сеть — нить А: baseline для всех модулей (§1.2).

Один класс обслуживает и единственный нейрон из E1.1, и раздутую сеть из E1.4,
и MLP по развёрнутым изображениям Pet из E1.5/E2.1. Разница — только конфиг.
"""

from __future__ import annotations

from torch import nn

from ..core.registry import register_model

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
}


@register_model("mlp")
def build_mlp(
    *,
    data_meta: dict,
    hidden: list[int] | None = None,
    activation: str = "relu",
    dropout: float = 0.0,
    flatten: bool = False,
    in_features: int | None = None,
    out_features: int | None = None,
) -> nn.Module:
    """hidden=[] или None -> один линейный слой, то есть ровно один нейрон
    на выход. Именно это нужно для E1.1.
    """
    in_dim = in_features or data_meta["in_features"]
    out_dim = out_features or data_meta["out_features"]

    layers: list[nn.Module] = []
    if flatten:
        layers.append(nn.Flatten())

    prev = in_dim
    for width in hidden or []:
        layers.append(nn.Linear(prev, width))
        layers.append(_ACTIVATIONS[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = width

    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)
