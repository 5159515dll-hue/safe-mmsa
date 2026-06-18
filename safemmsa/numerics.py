"""确定性数值工具（numpy，纯函数，无 torch）。

被三状态计算、晚融合主干、有界修正复用。所有"分布"在 EMOTION_LABELS 顺序上对齐。
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .labels import EMOTION_LABELS, NUM_CLASSES

_EPS = 1e-12
_LOG_K = float(np.log(NUM_CLASSES))  # ln(7)，熵归一化分母


def as_vector(distribution: Mapping[str, float]) -> np.ndarray:
    """dict 分布 → EMOTION_LABELS 顺序的 float 向量（缺失类按 0）。"""
    return np.array([float(distribution.get(label, 0.0)) for label in EMOTION_LABELS], dtype=float)


def as_distribution(vector: Sequence[float]) -> dict[str, float]:
    """向量 → dict 分布（EMOTION_LABELS 顺序）。"""
    return {label: float(vector[i]) for i, label in enumerate(EMOTION_LABELS)}


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """裁剪到非负并归一化为概率；全零/非正回退均匀分布。"""
    clipped = np.clip(np.asarray(vector, dtype=float), 0.0, None)
    total = float(clipped.sum())
    if total <= _EPS:
        return np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=float)
    return clipped / total


def normalize_distribution(distribution: Mapping[str, float]) -> dict[str, float]:
    """任意 dict → 裁剪到 7 类并归一化的概率 dict（空/非正→均匀）。"""
    return as_distribution(normalize_vector(as_vector(distribution)))


def softmax(logits: Sequence[float]) -> np.ndarray:
    """数值稳定 softmax。"""
    z = np.asarray(logits, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def logits_from_distribution(distribution: Mapping[str, float] | Sequence[float]) -> np.ndarray:
    """概率分布 → logits = log(p)（带下限），用于晚融合基线的 z⁰。

    经 softmax 还原时与原分布一致（在归一化意义下）。
    """
    if isinstance(distribution, Mapping):
        p = normalize_vector(as_vector(distribution))
    else:
        p = normalize_vector(np.asarray(distribution, dtype=float))
    return np.log(np.clip(p, _EPS, None))


def entropy(p: Sequence[float]) -> float:
    """自然对数香农熵 H(p) = -Σ p·ln p。"""
    arr = np.clip(np.asarray(p, dtype=float), _EPS, None)
    arr = arr / arr.sum()
    return float(-np.sum(arr * np.log(arr)))


def normalized_entropy(p: Sequence[float]) -> float:
    """归一化熵 H(p)/ln(K) ∈ [0,1]（均匀分布=1，one-hot=0）。"""
    return float(np.clip(entropy(p) / _LOG_K, 0.0, 1.0))


def max_confidence(p: Sequence[float]) -> float:
    """最大类概率 max_k p_k。"""
    return float(np.max(np.asarray(p, dtype=float)))


def top2_margin(p: Sequence[float]) -> float:
    """Top-1 与 Top-2 概率之差（≥0）。"""
    s = np.sort(np.asarray(p, dtype=float))[::-1]
    if s.size < 2:
        return float(s[0]) if s.size else 0.0
    return float(s[0] - s[1])


def top2_labels(distribution: Mapping[str, float]) -> tuple[str, str]:
    """返回概率最高的两类标签 (top1, top2)，按 EMOTION_LABELS 稳定排序打破并列。"""
    v = as_vector(distribution)
    order = sorted(range(NUM_CLASSES), key=lambda i: (-v[i], i))
    top1 = EMOTION_LABELS[order[0]]
    top2 = EMOTION_LABELS[order[1]]
    return top1, top2


def argmax_label(distribution: Mapping[str, float]) -> str:
    """概率最高的标签（稳定打破并列）。"""
    return top2_labels(distribution)[0]


def clip01(x: float) -> float:
    """裁剪到 [0,1]。"""
    return float(min(1.0, max(0.0, x)))


def sigmoid(x: float) -> float:
    """逻辑斯谛函数。"""
    return float(1.0 / (1.0 + np.exp(-x)))


def l2(vector: Sequence[float]) -> float:
    """L2 范数。"""
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))
