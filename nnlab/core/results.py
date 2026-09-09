"""Реестр запусков: §4.2 спецификации.

Главный артефакт проекта. Всё остальное — код, который его порождает,
и витрины, которые его читают.

Правило §5.7 реализовано здесь буквально: агрегатор отказывается усреднять
время по запускам с разным устройством. Сорок секунд на T4 и сорок секунд
на CPU — разные утверждения, и складывать их в одну ячейку нельзя.
"""

from __future__ import annotations

import json
import platform
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "results" / "runs"
SMOKE_DIR = REPO_ROOT / "results" / "_smoke"


@dataclass
class RunRecord:
    exp_id: str
    name: str
    seed: int
    config_hash: str
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    history: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def base_env() -> dict[str, Any]:
    import torch

    return {
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


def save(record: RunRecord, run_id: str, smoke: bool = False) -> Path:
    """Один запуск — один новый файл. Существующие файлы не переписываются
    другим прогоном, потому что имя содержит exp_id, хеш конфига и сид.
    """
    directory = SMOKE_DIR if smoke else RUNS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def load_all(directory: Path | None = None) -> list[dict]:
    directory = directory or RUNS_DIR
    if not directory.exists():
        return []
    out = []
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def aggregate(runs: list[dict]) -> list[dict]:
    """Группирует по (exp_id, config_hash), считает среднее и σ по сидам."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        groups[(run["exp_id"], run["config_hash"])].append(run)

    rows = []
    for (exp_id, cfg_hash), items in sorted(groups.items()):
        devices = {r["env"].get("device_name", "?") for r in items}
        metric_keys = sorted({k for r in items for k in r["metrics"]})

        agg_metrics = {}
        for key in metric_keys:
            values = [r["metrics"][key] for r in items if key in r["metrics"]]
            agg_metrics[key] = {
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }

        times = [r["train"].get("wall_time_s") for r in items]
        times = [t for t in times if t is not None]
        # §5.7: смешанные устройства -> время не усредняем
        time_mean = mean(times) if times and len(devices) == 1 else None

        rows.append(
            {
                "exp_id": exp_id,
                "config_hash": cfg_hash,
                "name": items[0]["name"],
                "n_seeds": len(items),
                "devices": sorted(devices),
                "mixed_devices": len(devices) > 1,
                "params_trainable": items[0]["model"].get("params_trainable"),
                "metrics": agg_metrics,
                "wall_time_s": time_mean,
            }
        )
    return rows


def to_markdown(rows: list[dict]) -> str:
    """Сводная таблица §8. Ни одно число в отчёте не появляется, если его нет здесь."""
    lines = [
        "# Сводная таблица результатов",
        "",
        "Генерируется автоматически: `python -m scripts.make_summary`. Руками не править.",
        "",
        "| ID | Модель | Сидов | Устройство | Параметры (обуч.) | Метрика | Значение (±σ) | Время, с |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        device = " + ".join(row["devices"])
        params = row["params_trainable"]
        params_str = f"{params:,}".replace(",", " ") if params else "—"
        time_str = (
            "разные устройства" if row["mixed_devices"] else _fmt(row["wall_time_s"], 1)
        )
        if not row["metrics"]:
            lines.append(
                f"| {row['exp_id']} | {row['name']} | {row['n_seeds']} | {device} | "
                f"{params_str} | — | — | {time_str} |"
            )
        first = True
        for key, stat in row["metrics"].items():
            value = f"{stat['mean']:.4f}"
            if stat["n"] > 1:
                value += f" ± {stat['std']:.4f}"
            lines.append(
                f"| {row['exp_id'] if first else ''} | {row['name'] if first else ''} | "
                f"{row['n_seeds'] if first else ''} | {device if first else ''} | "
                f"{params_str if first else ''} | {key} | {value} | "
                f"{time_str if first else ''} |"
            )
            first = False

    if any(r["mixed_devices"] for r in rows):
        lines += [
            "",
            "> Для строк, помеченных «разные устройства», время обучения не усредняется:",
            "> запуски выполнены на разном железе (§5.7). Это либо ошибка протокола,",
            "> либо сознательное сравнение CPU и GPU (E5.3) — разбирается отдельно.",
        ]
    return "\n".join(lines) + "\n"


def write_summary(path: Path | None = None) -> Path:
    path = path or (REPO_ROOT / "results" / "summary.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_markdown(aggregate(load_all())), encoding="utf-8")
    return path
