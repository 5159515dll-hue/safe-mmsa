"""模块6 — 融合增强注册表（插点）+ 有界修正数学（确定性，现在就实现）。

默认 ``identity_baseline``：p*=p⁰、修正幅度 0。真增强网络 D 训练后只产出 ``raw_adjustment``
（logit 空间残差），交给 ``apply_bounded_correction`` 做"由边界评分 B 门控的有界叠加"：
B≈0 → 几乎不改；B≈1 → 允许较大修正。这一段是确定性的，论文里属"融合增强方法"的可控部分。

``register_reliability_head`` 在此 re-export（实际消费在 states.py），归入模块6 插点命名空间。
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from . import numerics as N
from .contracts import (
    BoundaryState,
    CompetitionState,
    EnhancementResult,
    InitialPrediction,
    ReliabilityState,
)
# 归入模块6 插点命名空间（消费在 states.compute_reliability_state）。
from .states import get_reliability_head, register_reliability_head  # noqa: F401

EnhancementBackend = Callable[
    [InitialPrediction, BoundaryState, CompetitionState, ReliabilityState], EnhancementResult
]

_enhancement: Optional[EnhancementBackend] = None

# 修正幅度上界系数：B=1 时允许的最大 L2 修正量。
DEFAULT_ETA = 0.5


def register_enhancement(backend: Optional[EnhancementBackend]) -> None:
    """注册真实增强网络后端。传 None 恢复默认 identity 基线。"""
    global _enhancement
    _enhancement = backend


def get_enhancement() -> Optional[EnhancementBackend]:
    return _enhancement


def is_stub() -> bool:
    """当前是否在用 identity 基线（无真实增强网络）。供 /api/models/status 使用。"""
    return _enhancement is None


def apply_bounded_correction(
    p0: dict[str, float],
    raw_adjustment: Sequence[float],
    boundary_score: float,
    eta: float = DEFAULT_ETA,
) -> tuple[np.ndarray, float, bool]:
    """以 p⁰ 为锚做有界残差修正。

    raw_adjustment 视为 logit 空间残差：p_cand = softmax(log p⁰ + raw_adjustment)。
    令 Δ = p_cand − p⁰；上界 bound = eta·clip(B,0,1)。若 ‖Δ‖ > bound 则按比例缩回（bounded=True）。
    返回 (p_final 向量, 修正幅度 ‖p*−p⁰‖, 是否触发上界)。
    """
    p0_vec = N.normalize_vector(N.as_vector(p0))
    adj = np.asarray(raw_adjustment, dtype=float)
    if adj.shape[0] != p0_vec.shape[0]:
        raise ValueError("raw_adjustment 长度必须等于类别数")

    z0 = np.log(np.clip(p0_vec, 1e-12, None))
    p_cand = N.softmax(z0 + adj)
    delta = p_cand - p0_vec

    bound = eta * N.clip01(boundary_score)
    norm = float(np.linalg.norm(delta))
    bounded = False
    if norm > bound and norm > 1e-12:
        delta = delta * (bound / norm)
        bounded = True

    p_final = N.normalize_vector(p0_vec + delta)  # 裁剪负值后再归一化
    magnitude = float(np.linalg.norm(p_final - p0_vec))
    return p_final, magnitude, bounded


def build_enhancement_result(
    p_final: np.ndarray,
    magnitude: float,
    bounded: bool,
    name: str,
) -> EnhancementResult:
    """从有界修正结果构造 EnhancementResult（供真实后端复用，减少样板）。"""
    logits = np.log(np.clip(p_final, 1e-12, None))
    return EnhancementResult(
        final_distribution=N.as_distribution(p_final),
        final_logits=[float(x) for x in logits],
        correction_magnitude=round(magnitude, 6),
        bounded=bounded,
        enhancement=name,
    )


def identity_baseline(
    initial: InitialPrediction,
    boundary: BoundaryState,
    competition: CompetitionState,
    reliability: ReliabilityState,
) -> EnhancementResult:
    """默认增强：p*=p⁰、修正 0。框架就绪，真模型注册后自然产出非零修正。"""
    return EnhancementResult(
        final_distribution=dict(initial.distribution),
        final_logits=list(initial.logits),
        correction_magnitude=0.0,
        bounded=False,
        enhancement="identity_baseline",
    )


def enhance(
    initial: InitialPrediction,
    boundary: BoundaryState,
    competition: CompetitionState,
    reliability: ReliabilityState,
) -> EnhancementResult:
    """增强：有注册后端用真实增强网络，否则用 identity 基线。"""
    backend = _enhancement
    if backend is not None:
        return backend(initial, boundary, competition, reliability)
    return identity_baseline(initial, boundary, competition, reliability)
