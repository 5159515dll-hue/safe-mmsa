"""模块5 — 融合主干注册表（插点）+ 默认晚融合基线。

默认 ``late_weighted_baseline``（迁自 AIoT ``fuse_distribution`` 思路）：每个有分布的模态按
"自certainty × 模态先验"加权平均 → p⁰，z⁰=log(p⁰)。这恰好就是论文消融实验的 baseline。

训练完成后 ``register_fusion_backbone(backend)`` 挂载"跨模态注意力 + 低秩张量融合"主干
（输出 h⁰/z⁰/p⁰），其余代码零改动。
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from . import numerics as N
from .contracts import InitialPrediction, ModalityOutput
from .labels import EMOTION_LABELS, NUM_CLASSES

FusionBackboneBackend = Callable[[Sequence[ModalityOutput]], InitialPrediction]

_backbone: Optional[FusionBackboneBackend] = None

# 模态先验（在自certainty 之上再乘一层）。文本语义信息量略高。
_MODALITY_PRIOR: dict[str, float] = {"video": 1.0, "audio": 1.0, "text": 1.2}


def register_fusion_backbone(backend: Optional[FusionBackboneBackend]) -> None:
    """注册真实融合主干。传 None 恢复默认晚融合基线。"""
    global _backbone
    _backbone = backend


def get_fusion_backbone() -> Optional[FusionBackboneBackend]:
    return _backbone


def is_stub() -> bool:
    """当前是否在用默认基线（无真实主干）。供 /api/models/status 使用。"""
    return _backbone is None


def _certainty(p: np.ndarray) -> float:
    """单模态分布自certainty ∈ [0,1]：max_conf / (1−归一化熵) / top2_margin 的均值。"""
    return float((N.max_confidence(p) + (1.0 - N.normalized_entropy(p)) + N.top2_margin(p)) / 3.0)


def late_weighted_baseline(outputs: Sequence[ModalityOutput]) -> InitialPrediction:
    """晚融合基线：按自certainty×先验加权平均各模态分布 → p⁰。

    缺分布/低certainty 的模态自动降级（权重→0），不污染其余模态。全缺 → 中性兜底。
    """
    weighted = np.zeros(NUM_CLASSES)
    weight_total = 0.0
    for o in outputs:
        if not o.available or o.distribution is None:
            continue
        p = N.normalize_vector(N.as_vector(o.distribution))
        weight = _certainty(p) * _MODALITY_PRIOR.get(o.modality, 1.0)
        if weight <= 0:
            continue
        weighted += weight * p
        weight_total += weight

    if weight_total <= 0:
        # 全部缺失/零certainty → 中性兜底（neutral 略高、其余近均匀，保留不确定性 → B 偏高）。
        p0 = np.full(NUM_CLASSES, 0.7 / NUM_CLASSES)
        p0[EMOTION_LABELS.index("neutral")] += 0.3
        p0 = N.normalize_vector(p0)
    else:
        p0 = N.normalize_vector(weighted / weight_total)

    z0 = np.log(np.clip(p0, 1e-12, None))
    return InitialPrediction(
        distribution=N.as_distribution(p0),
        logits=[float(x) for x in z0],
        fused_feature_ref=None,
        backbone="late_weighted_baseline",
    )


def fuse(outputs: Sequence[ModalityOutput]) -> InitialPrediction:
    """融合：有注册主干用真实主干，否则用默认晚融合基线。"""
    backend = _backbone
    if backend is not None:
        return backend(outputs)
    return late_weighted_baseline(outputs)
