"""Конфиги: §4.1 п. 3 — смена датасета, модели или сида есть правка конфига, а не кода.

Намеренно без hydra. Нужны три вещи: загрузить yaml, слить с дефолтами,
посчитать хеш. Это сорок строк, а hydra — ещё одна зависимость, которую
пришлось бы синхронизировать между WSL и Colab.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"

# Поля, которые НЕ влияют на результат эксперимента и потому не входят в хеш.
# Сид исключён сознательно: три сида одной конфигурации обязаны иметь один
# config_hash, иначе results.py не сможет их сгруппировать (§4.2).
# `name` тоже косметика: переименование эксперимента не должно менять хеш,
# иначе одна правка подписи в yaml тихо расколет группу в summary.md надвое.
_HASH_EXCLUDE = {"seed", "smoke", "device", "num_workers", "resume", "notes", "name"}

DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "device": "auto",
    "amp": True,
    "smoke": False,
    "resume": True,
    "num_workers": 4,
    "train": {
        "epochs": 100,
        "batch_size": 32,
        "lr": 1e-3,
        "optimizer": "adam",
        "loss": "mse",
        "metric": "rmse",
        "metric_mode": "min",  # min для ошибок, max для accuracy/IoU
        "early_stopping": None,  # либо {"patience": 10, "min_delta": 0.0}
        "checkpoint_every": 1,  # в эпохах; для pix2pix будет в шагах
        "log_unit": "epoch",  # epoch | step (§4.2, история pix2pix пишется по шагам)
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def _coerce(text: str) -> Any:
    """`train.lr=0.05` из командной строки должно стать числом, а не строкой."""
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_config(exp_id: str, overrides: list[str] | None = None) -> dict:
    """Читает configs/exp/<exp_id>.yaml, сливает с DEFAULTS и переопределениями."""
    path = CONFIG_ROOT / "exp" / f"{exp_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Нет конфига эксперимента: {path}")

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = _deep_merge(DEFAULTS, raw)
    cfg.setdefault("exp_id", exp_id)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Переопределение должно быть вида key.sub=value, получено: {item}")
        key, value = item.split("=", 1)
        _set_by_path(cfg, key.strip(), _coerce(value.strip()))

    return cfg


def _strip_for_hash(cfg: dict) -> dict:
    return {
        k: (_strip_for_hash(v) if isinstance(v, dict) else v)
        for k, v in sorted(cfg.items())
        if k not in _HASH_EXCLUDE
    }


def config_hash(cfg: dict, length: int = 6) -> str:
    """Короткий стабильный хеш содержательной части конфига."""
    payload = json.dumps(_strip_for_hash(cfg), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def run_id(cfg: dict) -> str:
    """Имя прогона по правилу §4.2: exp_id__hash__seedN.

    Одно имя = один файл. Реестр append-only, поэтому WSL и Colab пушат
    результаты в одну ветку без конфликтов слияния.
    """
    return f"{cfg['exp_id']}__{config_hash(cfg)}__seed{cfg['seed']}"


def apply_smoke(cfg: dict) -> dict:
    """§4.1 п. 7: за десять секунд на ноутбуке проверить, что ничего не падает."""
    if not cfg.get("smoke"):
        return cfg
    cfg = copy.deepcopy(cfg)
    cfg["train"]["epochs"] = 2
    cfg["train"]["early_stopping"] = None
    cfg["train"]["limit_batches"] = 2
    cfg["num_workers"] = 0
    return cfg
