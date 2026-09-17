.PHONY: smoke summary test lint clean clean-ckpt

# Проверка, что каркас жив: два батча, две эпохи, десять секунд.
smoke:
	python -m scripts.run --exp E1.1 --smoke

summary:
	python -m scripts.make_summary

test:
	python -m pytest -q tests/

lint:
	python -m ruff check nnlab scripts tests

# checkpoints здесь НЕ трогаем: в них лежит возобновление оборванных прогонов (§4.1 п. 6).
clean:
	rm -rf results/_smoke
	find . -name __pycache__ -type d -exec rm -rf {} +

clean-ckpt:
	rm -rf checkpoints
