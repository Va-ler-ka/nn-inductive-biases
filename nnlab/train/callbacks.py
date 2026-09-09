"""Ранняя остановка и чекпоинты.

Чекпоинт здесь не удобство, а требование (§4.1 п. 6): сессия Colab может
оборваться в любой момент, и четырёхчасовой прогон pix2pix обязан продолжаться
с того места, где встал, а не начинаться заново.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..core import seeding
from ..core.metrics import is_better

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / "checkpoints"


class EarlyStopping:
    """§5.4: везде, где не мешает задаче, с восстановлением лучших весов."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1
        self.best_state: dict | None = None
        self.bad_epochs = 0
        self.stopped = False

    def step(self, value: float, epoch: int, model) -> bool:
        if is_better(self.mode, value, self.best, self.min_delta):
            self.best = value
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True
        return self.stopped

    def restore(self, model) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class Checkpointer:
    """Сохраняет всё, что нужно для бесшовного продолжения прогона."""

    def __init__(self, run_id: str, every: int = 1, root: Path | None = None):
        self.dir = (root or CKPT_DIR) / run_id
        self.every = max(1, every)
        self.path = self.dir / "last.pt"

    def exists(self) -> bool:
        return self.path.exists()

    def save(self, *, epoch: int, model, optimizer, scaler, history: dict, extra: dict | None = None) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "history": history,
            "rng": seeding.rng_state(),
        }
        if extra:
            payload.update(extra)
        tmp = self.path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(self.path)  # атомарно: обрыв во время записи не портит чекпоинт

    def load(self, model, optimizer, scaler, map_location) -> dict:
        payload = torch.load(self.path, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        seeding.load_rng_state(payload["rng"])
        return payload

    def should_save(self, epoch: int) -> bool:
        return epoch % self.every == 0

    def cleanup(self) -> None:
        """После успешного завершения чекпоинт больше не нужен: результат в json."""
        if self.path.exists():
            self.path.unlink()
        if self.dir.exists() and not any(self.dir.iterdir()):
            self.dir.rmdir()
