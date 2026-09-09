"""run_experiment — единственная точка входа (§4.1 п. 1).

Ни один эксперимент не запускается в обход этой функции. Всё, что она делает,
делается одинаково для всех шести лабораторных: сид, устройство, точность,
цикл, замеры, чекпоинт, запись json.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..core import config as cfg_mod
from ..core import device as dev_mod
from ..core import metrics as metrics_mod
from ..core import results as results_mod
from ..core import seeding
from ..core.registry import build_dataset, build_model, count_params
from .callbacks import Checkpointer, EarlyStopping

_LOSSES = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "ce": nn.CrossEntropyLoss,
    "bce": nn.BCEWithLogitsLoss,
}

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


def _resolve_workers(requested: int, dataset_size: int) -> int:
    """Не просим больше воркеров, чем есть ядер, и не плодим их на крошечных
    датасетах: на двенадцати точках накладные расходы больше самой работы.
    """
    cores = os.cpu_count() or 1
    if dataset_size < 512:
        return 0
    return max(0, min(requested, cores))


def _make_loader(dataset, cfg: dict, *, shuffle: bool) -> DataLoader | None:
    if dataset is None:
        return None
    batch_size = min(cfg["train"]["batch_size"], len(dataset))
    workers = _resolve_workers(cfg["num_workers"], len(dataset))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        worker_init_fn=seeding.worker_init_fn if workers > 0 else None,
        generator=seeding.make_generator(cfg["seed"]),
        drop_last=False,
        pin_memory=False,
    )


def _run_epoch(model, loader, loss_fn, optimizer, precision, scaler, limit_batches=None):
    """Одна эпоха. optimizer=None -> режим оценки."""
    training = optimizer is not None
    model.train(training)

    total_loss, total_n = 0.0, 0
    preds, targets = [], []

    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(loader):
            if limit_batches is not None and i >= limit_batches:
                break
            x = x.to(precision.device, non_blocking=True)
            y = y.to(precision.device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=precision.device.type,
                dtype=precision.dtype,
                enabled=precision.use_amp,
            ):
                out = model(x)
                loss = loss_fn(out, y)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_n = x.size(0)
            total_loss += loss.item() * batch_n
            total_n += batch_n
            preds.append(out.detach().float().cpu().numpy())
            targets.append(y.detach().cpu().numpy())

    return (
        total_loss / max(total_n, 1),
        np.concatenate(preds),
        np.concatenate(targets),
    )


def run_experiment(cfg: dict, *, verbose: bool = True) -> dict:
    cfg = cfg_mod.apply_smoke(cfg)
    run_id = cfg_mod.run_id(cfg)
    tcfg = cfg["train"]

    seeding.seed_everything(cfg["seed"])
    device = dev_mod.resolve_device(cfg["device"])
    precision = dev_mod.precision_policy(device, amp_allowed=cfg["amp"])
    dev_mod.reset_peak_memory(device)

    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))

    bundle = build_dataset(cfg)
    train_loader = _make_loader(bundle.train, cfg, shuffle=True)
    val_loader = _make_loader(bundle.val, cfg, shuffle=False)
    test_loader = _make_loader(bundle.test, cfg, shuffle=False)

    model = build_model(cfg, bundle.meta).to(device)
    loss_fn = _LOSSES[tcfg["loss"]]()
    optimizer = _OPTIMIZERS[tcfg["optimizer"]](model.parameters(), lr=tcfg["lr"])
    scaler = torch.amp.GradScaler(device.type) if precision.use_scaler else None

    metric_name = tcfg["metric"]
    metric_mode = tcfg["metric_mode"]

    history: dict[str, list] = {"x_unit": tcfg["log_unit"], "x": [], "loss": []}
    if val_loader is not None:
        history["val_loss"] = []
        history[f"val_{metric_name}"] = []
    history[f"train_{metric_name}"] = []

    # ---- возобновление (§4.1 п. 6) -------------------------------------------
    ckpt = Checkpointer(run_id, every=tcfg["checkpoint_every"])
    start_epoch, resumed_from = 0, 0
    if cfg["resume"] and ckpt.exists():
        payload = ckpt.load(model, optimizer, scaler, map_location=device)
        start_epoch = payload["epoch"]
        resumed_from = start_epoch
        history = payload["history"]
        if verbose:
            print(f"[{run_id}] продолжаем с эпохи {start_epoch}")

    stopper = None
    if tcfg["early_stopping"] and val_loader is not None:
        stopper = EarlyStopping(mode=metric_mode, **tcfg["early_stopping"])

    limit = tcfg.get("limit_batches")
    started = time.perf_counter()
    epoch = start_epoch

    for epoch in range(start_epoch + 1, tcfg["epochs"] + 1):
        train_loss, tr_pred, tr_true = _run_epoch(
            model, train_loader, loss_fn, optimizer, precision, scaler, limit
        )
        history["x"].append(epoch)
        history["loss"].append(train_loss)
        history[f"train_{metric_name}"].append(
            metrics_mod.compute(metric_name, tr_pred, tr_true)
        )

        if val_loader is not None:
            val_loss, va_pred, va_true = _run_epoch(
                model, val_loader, loss_fn, None, precision, None, limit
            )
            val_metric = metrics_mod.compute(metric_name, va_pred, va_true)
            history["val_loss"].append(val_loss)
            history[f"val_{metric_name}"].append(val_metric)
            if stopper is not None and stopper.step(val_metric, epoch, model):
                if verbose:
                    print(f"[{run_id}] ранняя остановка на эпохе {epoch}")
                break

        if ckpt.should_save(epoch):
            ckpt.save(
                epoch=epoch, model=model, optimizer=optimizer,
                scaler=scaler, history=history,
            )

    if stopper is not None:
        stopper.restore(model)

    wall_time = time.perf_counter() - started

    # ---- финальная оценка -----------------------------------------------------
    final: dict[str, float] = {}
    for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        if loader is None:
            continue
        loss_value, pred, true = _run_epoch(model, loader, loss_fn, None, precision, None, limit)
        final[f"{split}_loss"] = loss_value
        final[f"{split}_{metric_name}"] = metrics_mod.compute(metric_name, pred, true)

    record = results_mod.RunRecord(
        exp_id=cfg["exp_id"],
        name=cfg.get("name", cfg["exp_id"]),
        seed=cfg["seed"],
        config_hash=cfg_mod.config_hash(cfg),
        data={"name": cfg["data"]["name"], **bundle.sizes()},
        model={"name": cfg["model"]["name"], **count_params(model)},
        train={
            "epochs_run": epoch,
            "epochs_planned": tcfg["epochs"],
            "early_stopped": bool(stopper and stopper.stopped),
            "best_epoch": stopper.best_epoch if stopper else None,
            "wall_time_s": round(wall_time, 2),
            "wall_time_per_epoch_s": round(wall_time / max(epoch - start_epoch, 1), 3),
            "peak_mem_mb": dev_mod.peak_memory_mb(device),
            "resumed_from_step": resumed_from,
        },
        metrics=final,
        history=history,
        env={
            **results_mod.base_env(),
            "device": device.type,
            "device_name": dev_mod.device_name(device),
            "host": dev_mod.detect_host(),
            "precision": precision.label,
            "num_workers": _resolve_workers(cfg["num_workers"], len(bundle.train)),
        },
        config=cfg,
    )

    path = results_mod.save(record, run_id, smoke=cfg["smoke"])
    if not cfg["smoke"]:
        ckpt.cleanup()

    if verbose:
        summary = "  ".join(f"{k}={v:.4f}" for k, v in final.items())
        print(f"[{run_id}] {summary}  ({wall_time:.1f} c, {precision.label}, {device.type})")
        print(f"    -> {Path(path).relative_to(results_mod.REPO_ROOT)}")

    return record.to_dict()
