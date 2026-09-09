"""Реестр моделей и датасетов по имени.

Смысл — в §4.3 спецификации: добавление новой архитектуры позже сводится
к одному файлу в models/ плюс конфиг, ядро не трогается.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from torch.utils.data import Dataset

_DATASETS: dict[str, Callable[..., "DataBundle"]] = {}
_MODELS: dict[str, Callable[..., Any]] = {}


@dataclass
class DataBundle:
    """Что датасет отдаёт движку.

    val и test могут быть None: в E1.1 (один нейрон на двенадцати точках)
    валидации нет и выдумывать её не надо.
    """

    train: Dataset
    val: Dataset | None = None
    test: Dataset | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def sizes(self) -> dict[str, int]:
        out = {"n_train": len(self.train)}
        if self.val is not None:
            out["n_val"] = len(self.val)
        if self.test is not None:
            out["n_test"] = len(self.test)
        return out


def register_dataset(name: str):
    def deco(fn):
        if name in _DATASETS:
            raise KeyError(f"Датасет '{name}' уже зарегистрирован")
        _DATASETS[name] = fn
        return fn

    return deco


def register_model(name: str):
    def deco(fn):
        if name in _MODELS:
            raise KeyError(f"Модель '{name}' уже зарегистрирована")
        _MODELS[name] = fn
        return fn

    return deco


def build_dataset(cfg: dict) -> DataBundle:
    spec = dict(cfg["data"])
    name = spec.pop("name")
    if name not in _DATASETS:
        raise KeyError(f"Неизвестный датасет '{name}'. Есть: {sorted(_DATASETS)}")
    return _DATASETS[name](**spec)


def build_model(cfg: dict, data_meta: dict):
    spec = dict(cfg["model"])
    name = spec.pop("name")
    if name not in _MODELS:
        raise KeyError(f"Неизвестная модель '{name}'. Есть: {sorted(_MODELS)}")
    return _MODELS[name](data_meta=data_meta, **spec)


def count_params(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable}
