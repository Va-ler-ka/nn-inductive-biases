# Индуктивные смещения нейросетевых архитектур

Курсовой проект: шесть лабораторных работ курса собраны в один модульный проект.
Задание и все решения — в [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

**Исследовательский вопрос:** насколько выигрывает специализированная архитектура
у универсального полносвязного персептрона на данных с явной структурой, и за счёт чего.

## Установка

Ноутбук / WSL2, только CPU:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple
pip install -e .
```

Colab: см. `colab/run_on_colab.ipynb` — torch там не переустанавливается.

## Запуск

```bash
make smoke                                   # проверка каркаса, ~10 секунд
python -m scripts.run --exp E1.1 --seed 0 1 2
python -m scripts.run --exp E1.2 --set train.lr=0.1
make summary                                 # results/runs/*.json -> results/summary.md
make test
```

Ни один эксперимент не запускается в обход `run_experiment` (§4.1). Результат
прогона — json в `results/runs/`, а не число в выводе ячейки.

## Устройство и точность

Выбираются автоматически (`nnlab/core/device.py`), в конфигах железа нет:

| Где | Устройство | Точность |
|---|---|---|
| Ноутбук (Ryzen 7 5700U), WSL2 | CPU | fp32 |
| Colab, T4 | CUDA sm_75 | fp16 + GradScaler |
| Colab, A100 / L4 | CUDA | bf16 |

Любой эксперимент обязан запускаться на CPU — пусть медленно. Витрины в `defence/`
не требуют ни GPU, ни интернета: они читают json и рисуют.

## Состояние

Готово ядро (M0) и первые эксперименты M1. Порядок работ — §10 спецификации.
