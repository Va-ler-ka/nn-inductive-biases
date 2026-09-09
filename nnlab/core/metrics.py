"""Метрики поверх numpy (§4.3).

History — обычный dict[str, list[float]], а не объект фреймворка. Визуализация
получает numpy. Благодаря этому витрины не зависят ни от torch, ни от GPU:
они читают json и рисуют.
"""

from __future__ import annotations

import numpy as np

_REGISTRY: dict[str, callable] = {}


def register_metric(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn

    return deco


@register_metric("rmse")
def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred.ravel() - y_true.ravel()) ** 2)))


@register_metric("mae")
def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred.ravel() - y_true.ravel())))


@register_metric("accuracy")
def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    labels = y_pred.argmax(axis=1) if y_pred.ndim > 1 else (y_pred > 0.5).astype(int)
    return float((labels.ravel() == y_true.ravel()).mean())


def compute(name: str, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if name not in _REGISTRY:
        raise KeyError(f"Неизвестная метрика '{name}'. Есть: {sorted(_REGISTRY)}")
    return _REGISTRY[name](y_pred, y_true)


def is_better(name_mode: str, new: float, best: float, min_delta: float = 0.0) -> bool:
    if name_mode == "min":
        return new < best - min_delta
    return new > best + min_delta
