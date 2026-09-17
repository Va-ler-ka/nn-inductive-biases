.PHONY: smoke summary test lint clean clean-ckpt

# В Ubuntu/WSL исполняемого файла `python` нет, есть только `python3`, а `make`
# запускает каждую строку рецепта в отдельном шелле, где venv может быть не
# активирован. Поэтому интерпретатор выбирается явно: venv репозитория, если он
# есть (WSL), иначе системный python3 (Colab). Переопределяется из командной
# строки: make test PYTHON=python3.12
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

# Проверка, что каркас жив: два батча, две эпохи, десять секунд.
smoke:
	$(PYTHON) -m scripts.run --exp E1.1 --smoke

summary:
	$(PYTHON) -m scripts.make_summary

test:
	$(PYTHON) -m pytest -q tests/

lint:
	$(PYTHON) -m ruff check nnlab scripts tests

# checkpoints здесь НЕ трогаем: в них лежит возобновление оборванных прогонов (§4.1 п. 6).
clean:
	rm -rf results/_smoke
	find . -name __pycache__ -type d -exec rm -rf {} +

clean-ckpt:
	rm -rf checkpoints
