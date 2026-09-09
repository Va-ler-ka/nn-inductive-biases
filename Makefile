.PHONY: smoke summary test clean

# Проверка, что каркас жив: два батча, две эпохи, десять секунд.
smoke:
	python -m scripts.run --exp E1.1 --smoke

summary:
	python -m scripts.make_summary

test:
	python -m pytest -q tests/

clean:
	rm -rf results/_smoke checkpoints
	find . -name __pycache__ -type d -exec rm -rf {} +
