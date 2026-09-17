"""Кривые и таблицы для витрин.

Важное свойство: этот модуль НЕ импортирует torch. Витрина открывается на
ноутбуке за секунды, без GPU и без интернета, потому что читает json (§13.3).
Если сюда когда-нибудь просочится `import torch` — это баг.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .results import load_all


def runs_for(exp_id: str, runs: list[dict] | None = None) -> list[dict]:
    return [r for r in (runs or load_all()) if r["exp_id"] == exp_id]


def plot_history(
    exp_ids: str | list[str],
    key: str = "loss",
    *,
    runs: list[dict] | None = None,
    ax=None,
    logy: bool = False,
):
    """Кривые обучения для одного или нескольких экспериментов.

    Разные сиды одного эксперимента рисуются одним цветом: разброс видно,
    а легенда не разрастается.
    """
    exp_ids = [exp_ids] if isinstance(exp_ids, str) else exp_ids
    runs = runs or load_all()
    ax = ax or plt.subplots(figsize=(7, 4))[1]

    for i, exp_id in enumerate(exp_ids):
        color = f"C{i}"
        selected = runs_for(exp_id, runs)
        if not selected:
            continue
        for j, run in enumerate(selected):
            history = run["history"]
            if key not in history:
                continue
            ax.plot(
                history["x"], history[key],
                color=color, alpha=0.85,
                label=f"{exp_id} — {run['name']}" if j == 0 else None,
            )

    unit = (runs_for(exp_ids[0], runs) or [{"history": {"x_unit": "epoch"}}])[0]["history"]["x_unit"]
    ax.set_xlabel("эпоха" if unit == "epoch" else "шаг")
    ax.set_ylabel(key)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def compare_table(exp_ids: list[str], metric: str, runs: list[dict] | None = None) -> str:
    """Markdown-таблица «эксперимент → метрика ± σ → время», для витрины."""
    from statistics import mean, pstdev

    runs = runs or load_all()
    lines = ["| ID | Модель | Сидов | " + metric + " (±σ) | Время, с | Устройство |",
             "|---|---|---|---|---|---|"]
    for exp_id in exp_ids:
        selected = runs_for(exp_id, runs)
        if not selected:
            lines.append(f"| {exp_id} | — | 0 | нет запусков | — | — |")
            continue
        values = [r["metrics"][metric] for r in selected if metric in r["metrics"]]
        times = [r["train"]["wall_time_s"] for r in selected]
        # device_name, а не device: «cpu» на ноутбуке и «cpu» на Colab — разное железо,
        # и сводная таблица (results.aggregate) различает их именно так. Витрина
        # обязана показывать то же самое, иначе она тихо противоречит summary.md.
        devices = {r["env"].get("device_name", r["env"]["device"]) for r in selected}
        precisions = {r["env"].get("precision", "?") for r in selected}
        spread = f" ± {pstdev(values):.4f}" if len(values) > 1 else ""
        if len(values) > 1 and len(precisions) > 1:
            spread += " ⚠ разная точность"
        time_cell = f"{mean(times):.1f}" if len(devices) == 1 else "разные устройства"
        lines.append(
            f"| {exp_id} | {selected[0]['name']} | {len(selected)} | "
            f"{mean(values):.4f}{spread} | {time_cell} | {'+'.join(sorted(devices))} |"
        )
    return "\n".join(lines)
