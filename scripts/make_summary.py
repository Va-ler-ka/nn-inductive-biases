"""results/runs/*.json -> results/summary.md (§8).

Запускается на ноутбуке после git pull. Ни torch, ни GPU не требует.
"""

from nnlab.core.results import write_summary

if __name__ == "__main__":
    print(f"сводка -> {write_summary()}")
