"""Устройство и точность: §4.1 п. 5 и §4.4 спецификации.

Единственное место в проекте, где вообще упоминается конкретное железо.
Ни в моделях, ни в данных, ни в витринах ничего про cuda быть не должно.

Политика точности (§4.4):
    CUDA + bf16 есть   -> bf16, без скейлера   (Colab с A100 / L4)
    CUDA, bf16 нет     -> fp16, СО скейлером   (T4, sm_75 — основной случай)
    CPU                -> fp32, без скейлера   (ноутбук; Zen 2 не умеет AVX512-BF16)

Ветки под ROCm нет сознательно: интегрированная Radeon (Vega/gfx90c в 5700U)
официально не поддерживается ни ROCm, ни ROCm-on-WSL. См. §11.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Precision:
    device: torch.device
    dtype: torch.dtype  # dtype для autocast
    use_amp: bool
    use_scaler: bool
    label: str  # то, что уедет в env.precision записи о запуске

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"


def resolve_device(requested: str = "auto") -> torch.device:
    """`auto` -> cuda, если есть, иначе cpu. Явное значение уважаем как есть."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "В конфиге запрошен device=cuda, но CUDA недоступна. "
            "Либо запусти на Colab, либо поставь device=auto."
        )
    return torch.device(requested)


def precision_policy(device: torch.device, amp_allowed: bool = True) -> Precision:
    """Выбирает точность по устройству.

    Конфиг может только ЗАПРЕТИТЬ amp (`amp: false`), но не может потребовать
    dtype, которого на устройстве нет.
    """
    if device.type != "cuda" or not amp_allowed:
        return Precision(device, torch.float32, False, False, "fp32")

    if torch.cuda.is_bf16_supported():
        return Precision(device, torch.bfloat16, True, False, "bf16")

    # T4 / V100: тензорные ядра под fp16 есть, bf16 и TF32 нет -> скейлер обязателен
    return Precision(device, torch.float16, True, True, "fp16")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "cpu"


def peak_memory_mb(device: torch.device) -> float | None:
    """Пиковая память. На CPU честно возвращаем None, а не ноль."""
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
    return None


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def detect_host() -> str:
    """`colab` или `wsl` — попадает в env каждого запуска (§4.2)."""
    try:
        import google.colab  # noqa: F401

        return "colab"
    except ImportError:
        pass
    try:
        with open("/proc/version") as f:
            if "microsoft" in f.read().lower():
                return "wsl"
    except OSError:
        pass
    return "local"
