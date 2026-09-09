"""Детерминизм: §4.4 спецификации.

Полного детерминизма на GPU не добиваемся сознательно — вместо этого три сида
и разброс. Но всё, что можно зафиксировать дёшево, фиксируем.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Фиксирует ГСЧ во всех местах, где он есть."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Каждый воркер DataLoader получает свой воспроизводимый сид."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def make_generator(seed: int) -> torch.Generator:
    """Генератор для DataLoader: перемешивание тоже должно быть воспроизводимым."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def rng_state() -> dict:
    """Снимок состояния ГСЧ для чекпоинта (§4.1, п. 6)."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: dict) -> None:
    """Восстановление состояния ГСЧ при возобновлении прогона."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
