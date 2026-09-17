"""Тесты на инварианты ядра. Гоняются на ноутбуке за секунды: make test."""

from __future__ import annotations

import torch

import nnlab.data  # noqa: F401
import nnlab.models  # noqa: F401
from nnlab.core.config import config_hash, load_config, run_id
from nnlab.core.device import precision_policy, resolve_device
from nnlab.core.results import aggregate


def test_hash_ignores_seed():
    """Три сида одной конфигурации обязаны иметь один config_hash,
    иначе results.py не сможет их сгруппировать (§4.2)."""
    a = load_config("E1.1", ["seed=0"])
    b = load_config("E1.1", ["seed=7"])
    assert config_hash(a) == config_hash(b)
    assert run_id(a) != run_id(b)


def test_hash_reacts_to_content():
    a = load_config("E1.1")
    b = load_config("E1.1", ["train.lr=0.5"])
    assert config_hash(a) != config_hash(b)


def test_cpu_policy_is_fp32():
    """Zen 2 не умеет AVX512-BF16; на CPU честный fp32 (§4.4)."""
    policy = precision_policy(torch.device("cpu"))
    assert policy.label == "fp32"
    assert not policy.use_amp and not policy.use_scaler


def test_cuda_without_bf16_requires_scaler():
    """T4 и V100: fp16 обязательно со скейлером."""
    device = resolve_device("auto")
    if device.type != "cuda":
        return
    policy = precision_policy(device)
    if policy.label == "fp16":
        assert policy.use_scaler


def test_aggregate_refuses_mixed_devices():
    """§5.7: время по разным устройствам не усредняется."""

    def fake(device_name: str, value: float):
        return {
            "exp_id": "X1", "config_hash": "aaa", "name": "x", "seed": 0,
            "metrics": {"rmse": value},
            "train": {"wall_time_s": 10.0},
            "model": {"params_trainable": 1},
            "env": {"device_name": device_name},
        }

    same = aggregate([fake("T4", 1.0), fake("T4", 2.0)])[0]
    assert same["wall_time_s"] == 10.0
    assert not same["mixed_devices"]

    mixed = aggregate([fake("T4", 1.0), fake("AMD Ryzen 7 5700U", 2.0)])[0]
    assert mixed["wall_time_s"] is None
    assert mixed["mixed_devices"]


def test_viz_does_not_import_torch():
    """Витрины обязаны работать без torch и без GPU (§13.3)."""
    import ast
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    source = (repo_root / "nnlab" / "core" / "viz.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported


def test_aggregate_flags_mixed_precision():
    """§4.4: fp16 и fp32 расходятся в метриках, усреднять их молча нельзя."""

    def fake(precision: str, value: float):
        return {
            "exp_id": "X2", "config_hash": "bbb", "name": "x", "seed": 0,
            "metrics": {"rmse": value},
            "train": {"wall_time_s": 10.0},
            "model": {"params_trainable": 1},
            "env": {"device_name": "T4", "precision": precision},
        }

    same = aggregate([fake("fp16", 1.0), fake("fp16", 2.0)])[0]
    assert not same["mixed_precision"]

    mixed = aggregate([fake("fp16", 1.0), fake("fp32", 2.0)])[0]
    assert mixed["mixed_precision"]


def test_gitignore_does_not_hide_source():
    """`data/` без ведущего слэша выкидывает из репозитория nnlab/data/."""
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    rules = repo_root.joinpath(".gitignore").read_text(encoding="utf-8").splitlines()
    bad = [r for r in rules if r.strip() in {"data/", "data", "models/", "train/", "core/"}]
    assert not bad, f"неякорённые правила скроют код внутри nnlab/: {bad}"


def test_hash_ignores_cosmetic_name():
    """Переименование эксперимента не должно раскалывать группу в summary.md."""
    a = load_config("E1.1")
    b = load_config("E1.1", ["name=совсем_другая_подпись"])
    assert config_hash(a) == config_hash(b)
