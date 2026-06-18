"""消融：开关三状态/增强模块跑对照。

默认 identity 时 final==initial（delta≈0），框架就绪——真增强网络注册后，
关掉它（no_enhancement）即可量化其贡献。本模块通过临时改写 register_* 实现对照，
跑完恢复原状。
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

from safemmsa import backbone, enhancement, encoders
from safemmsa.enhancement import identity_baseline
from safemmsa.states import get_reliability_head, register_reliability_head

from .runner import EvalItem, run_manifest
from .schemas import EvalRunResult


@contextmanager
def _snapshot_registries():
    """保存并在退出时恢复所有 register_* 状态。"""
    saved = {
        "video": encoders.get_encoder("video"),
        "audio": encoders.get_encoder("audio"),
        "text": encoders.get_encoder("text"),
        "backbone": backbone.get_fusion_backbone(),
        "enhancement": enhancement.get_enhancement(),
        "reliability_head": get_reliability_head(),
    }
    try:
        yield
    finally:
        encoders.register_encoder("video", saved["video"])
        encoders.register_encoder("audio", saved["audio"])
        encoders.register_encoder("text", saved["text"])
        backbone.register_fusion_backbone(saved["backbone"])
        enhancement.register_enhancement(saved["enhancement"])
        register_reliability_head(saved["reliability_head"])


# 可用消融变体：name -> 在 snapshot 内施加的改写。
def _apply_variant(name: str) -> None:
    if name == "full":
        return
    if name == "no_enhancement":
        # 强制 identity：等价于关掉增强网络 D（final==initial）。
        enhancement.register_enhancement(
            lambda initial, b, c, r: identity_baseline(initial, b, c, r)
        )
        return
    if name == "no_reliability_head":
        register_reliability_head(None)
        return
    raise ValueError(f"unknown ablation variant: {name}")


def run_ablation(
    items: Sequence[EvalItem],
    variants: Sequence[str] = ("full", "no_enhancement"),
) -> dict[str, EvalRunResult]:
    """对每个变体在隔离的注册状态下跑同一清单，返回 {variant: 结果}。"""
    results: dict[str, EvalRunResult] = {}
    for name in variants:
        with _snapshot_registries():
            _apply_variant(name)
            results[name] = run_manifest(
                items, run_id=f"ablation_{name}", config={"ablation": name}
            )
    return results
