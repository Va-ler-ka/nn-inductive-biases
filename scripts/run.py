"""Единственный способ запустить эксперимент.

    python -m scripts.run --exp E1.1
    python -m scripts.run --exp E1.1 --seed 0 1 2
    python -m scripts.run --exp E1.1 --smoke
    python -m scripts.run --exp E1.2 --set train.lr=0.1 train.epochs=500

Из витрин ничего не запускается (§4.1 п. 4). Из Colab вызывается ровно эта
же команда — именно поэтому окружения обязаны совпадать по коду, а не по железу.
"""

from __future__ import annotations

import argparse

import nnlab.data  # noqa: F401  регистрация датасетов
import nnlab.models  # noqa: F401  регистрация моделей
from nnlab.core.config import load_config
from nnlab.core.results import write_summary
from nnlab.train.engine import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск эксперимента из конфига")
    parser.add_argument("--exp", required=True, help="ID эксперимента, например E1.1")
    parser.add_argument("--seed", type=int, nargs="+", default=None, help="один или несколько сидов")
    parser.add_argument("--smoke", action="store_true", help="проверочный прогон на двух батчах")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda")
    parser.add_argument("--no-resume", action="store_true", help="начать заново, игнорируя чекпоинт")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                        help="переопределения, например train.lr=0.05")
    parser.add_argument("--summary", action="store_true", help="пересобрать results/summary.md")
    args = parser.parse_args()

    base_overrides = list(args.set)
    if args.device:
        base_overrides.append(f"device={args.device}")
    if args.smoke:
        base_overrides.append("smoke=true")
    if args.no_resume:
        base_overrides.append("resume=false")

    seeds = args.seed if args.seed is not None else [None]
    for seed in seeds:
        overrides = list(base_overrides)
        if seed is not None:
            overrides.append(f"seed={seed}")
        cfg = load_config(args.exp, overrides)
        run_experiment(cfg)

    if args.summary and not args.smoke:
        print(f"сводка -> {write_summary()}")


if __name__ == "__main__":
    main()
