"""Синтетические данные модуля M1 (лабораторная 1).

E1.1 использует ровно те двенадцать точек, что в методичке: это позволяет
воспроизвести описанные там проблемы обучения, а не рассуждать о них абстрактно.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from ..core.registry import DataBundle, register_dataset

# §1.4 методички лабораторной 1, дословно
_LAB1_FEATURES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
_LAB1_LABELS = [5.0, 8.8, 9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2]


def _to_dataset(x: np.ndarray, y: np.ndarray) -> Dataset:
    return TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


@register_dataset("lab1_linreg")
def lab1_linreg(
    normalize: bool = False,
    normalize_target: bool = False,
    val_split: float = 0.0,
    seed: int = 0,
) -> DataBundle:
    """Двенадцать точек из методички.

    `normalize` масштабирует ТОЛЬКО признаки. Это не мелочь: если заодно
    нормировать цель, RMSE начнёт считаться в единицах стандартного отклонения
    и станет несопоставим с RMSE ненормированного прогона. Тогда E1.1 и E1.2
    нельзя положить в одну таблицу — а весь смысл задания 2 именно в сравнении
    с заданием 1. Цель нормируем только по явному требованию.
    """
    x = np.array(_LAB1_FEATURES, dtype=np.float32).reshape(-1, 1)
    y = np.array(_LAB1_LABELS, dtype=np.float32).reshape(-1, 1)

    meta = {"in_features": 1, "out_features": 1, "task": "regression"}

    if normalize:
        x_mean, x_std = x.mean(), x.std()
        x = (x - x_mean) / x_std
        meta["x_norm"] = {"mean": float(x_mean), "std": float(x_std)}

    if normalize_target:
        y_mean, y_std = y.mean(), y.std()
        y = (y - y_mean) / y_std
        meta["y_norm"] = {"mean": float(y_mean), "std": float(y_std)}
        meta["metric_units"] = "normalized"  # предупреждение для сводной таблицы

    if val_split <= 0:
        return DataBundle(train=_to_dataset(x, y), meta=meta)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_val = max(1, int(round(len(x) * val_split)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return DataBundle(
        train=_to_dataset(x[train_idx], y[train_idx]),
        val=_to_dataset(x[val_idx], y[val_idx]),
        meta=meta,
    )
