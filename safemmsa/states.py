"""模块3 — 三状态计算（论文创新点核心，确定性，无训练、无 torch）。

给定初始预测 p⁰ 与各模态单模态输出，确定性地算出三个样本级状态：
- 边界评分 B：样本离决策边界多近（越近越该被增强修正）。
- 类别竞争状态 C：Top-2 类别上各模态的方向支持，供增强网络 D 作输入。
- 模态可靠性 R：质量 + 单模态确定性 + 跨模态一致性 → 每模态可靠性 ∈ [0,1]。

这三者**驱动**后续的有界融合增强（enhancement.py）。本模块为论文核心，配高覆盖单测。
其中 R 的"每模态可靠性融合映射"留 ``register_reliability_head`` 后插（默认确定性线性聚合）。
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from . import numerics as N
from .contracts import (
    MODALITIES,
    BoundaryState,
    CompetitionState,
    ModalityOutput,
    ReliabilityState,
)
from .labels import EMOTION_LABELS, index_of

# ── 模块6 插点（可选可学习可靠性融合头）────────────────────────────
# head(quality_part, uncertainty_part, consistency_pairwise) -> {modality: reliability∈[0,1]}
ReliabilityHead = Callable[
    [dict[str, float], dict[str, float], dict[str, float]], dict[str, float]
]
_reliability_head: Optional[ReliabilityHead] = None


def register_reliability_head(head: Optional[ReliabilityHead]) -> None:
    """注册可学习的可靠性融合头（覆盖默认线性聚合）。传 None 恢复默认。"""
    global _reliability_head
    _reliability_head = head


def get_reliability_head() -> Optional[ReliabilityHead]:
    return _reliability_head


# ── 状态1：边界评分 B ──────────────────────────────────────────────
def compute_boundary_state(
    p0: Mapping[str, float],
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> BoundaryState:
    """边界评分 B ∈ [0,1]：三统计量加权后裁剪。

    - max_conf  = max_k p⁰_k
    - top2_margin = p⁰_(1) − p⁰_(2)
    - norm_entropy = H(p⁰)/ln(K)
    - B = clip( w1·(1−max_conf) + w2·(1−margin) + w3·norm_entropy , 0, 1)

    三项都是"离边界越近 → 越大"：低置信、Top-2 接近、高熵 ⇒ B 大 ⇒ 允许更大修正。
    """
    v = N.as_vector(p0)
    max_conf = N.max_confidence(v)
    margin = N.top2_margin(v)
    norm_ent = N.normalized_entropy(v)
    w1, w2, w3 = weights
    score = N.clip01(w1 * (1.0 - max_conf) + w2 * (1.0 - margin) + w3 * norm_ent)
    return BoundaryState(
        max_confidence=max_conf,
        top2_margin=margin,
        norm_entropy=norm_ent,
        score=score,
    )


# ── 状态2：类别竞争 C ──────────────────────────────────────────────
def compute_competition_state(
    p0: Mapping[str, float],
    unimodal_logits: Mapping[str, Sequence[float]],
) -> CompetitionState:
    """类别竞争状态：Top-2 类别上各模态的方向支持。

    - (top1, top2) = p⁰ 的前二
    - 每模态 dir_m = logit_m[top1] − logit_m[top2]（>0 偏向 top1，<0 偏向 top2）
    - margin = p⁰[top1] − p⁰[top2]；entropy = 归一化熵
    - vector = [dir_video, dir_audio, dir_text, margin, entropy]（缺模态方向记 0）
    """
    top1, top2 = N.top2_labels(p0)
    i1, i2 = index_of(top1), index_of(top2)

    per_modality_direction: dict[str, float] = {}
    for modality, logits in unimodal_logits.items():
        if logits is None:
            continue
        arr = list(logits)
        if len(arr) <= max(i1, i2):
            continue
        per_modality_direction[modality] = float(arr[i1] - arr[i2])

    v = N.as_vector(p0)
    margin = float(v[i1] - v[i2])
    entropy = N.normalized_entropy(v)

    # 固定长度 5 的状态向量（模态顺序 video,audio,text；缺模态记 0）。
    vector = [per_modality_direction.get(m, 0.0) for m in MODALITIES]
    vector.extend([margin, entropy])

    return CompetitionState(
        top1=top1,
        top2=top2,
        per_modality_direction=per_modality_direction,
        margin=margin,
        entropy=entropy,
        vector=vector,
    )


# ── 状态3：模态可靠性 R ────────────────────────────────────────────
def _quality_goodness(modality: str, q) -> Optional[float]:
    """把单模态信号质量指标聚合成 [0,1] 的"质量优度"；无任一可测项返回 None。"""
    scores: list[float] = []

    def add(value: Optional[float], goodness):
        if value is not None:
            scores.append(N.clip01(goodness(value)))

    if modality == "video":
        add(q.face_detection_rate, lambda x: x)
        add(q.frame_loss_rate, lambda x: 1.0 - x)
        add(q.blur, lambda x: x / (x + 100.0))                       # Laplacian 方差，越大越清晰（饱和）
        add(q.brightness, lambda x: 1.0 - 2.0 * abs((x / 255.0 if x > 1.0 else x) - 0.5))  # 居中最佳
    elif modality == "audio":
        add(q.snr_db, lambda x: (x + 5.0) / 35.0)                    # -5dB→0, 30dB→1
        add(q.voiced_ratio, lambda x: x)
        add(q.silence_ratio, lambda x: 1.0 - x)
    elif modality == "text":
        add(q.text_length, lambda x: x / (x + 10.0))                 # 越长越稳（饱和）
        add(q.valid_token_ratio, lambda x: x)
        add(q.low_freq_subword_ratio, lambda x: 1.0 - x)            # 低频子词越多越不可靠

    if not scores:
        return None
    return float(sum(scores) / len(scores))


def _self_certainty(output: ModalityOutput) -> Optional[float]:
    """单模态自certainty ∈ [0,1]：由其分布的 max_conf / (1−归一化熵) / top2_margin 平均。

    无分布（桩、无真实模型）→ None。
    """
    if output.distribution is None:
        return None
    v = N.as_vector(output.distribution)
    return float((N.max_confidence(v) + (1.0 - N.normalized_entropy(v)) + N.top2_margin(v)) / 3.0)


def _pair_consistency(
    a: ModalityOutput,
    b: ModalityOutput,
    competition: Optional[CompetitionState],
) -> float:
    """模态对一致性 ∈ [0,1]。

    优先用竞争类别对上的方向一致性（需双方有 logits 且 competition 给定）；
    退化为整分布的 Bhattacharyya 重叠；任一方无分布 → 0.5（未知）。
    """
    if (
        competition is not None
        and a.logits is not None
        and b.logits is not None
    ):
        i1, i2 = index_of(competition.top1), index_of(competition.top2)
        if max(i1, i2) < len(a.logits) and max(i1, i2) < len(b.logits):
            da = float(a.logits[i1] - a.logits[i2])
            db = float(b.logits[i1] - b.logits[i2])
            # tanh 压到 [-1,1] 再取符号/幅度一致性 → [0,1]
            return float((1.0 + np.tanh(da) * np.tanh(db)) / 2.0)

    if a.distribution is not None and b.distribution is not None:
        pa = N.normalize_vector(N.as_vector(a.distribution))
        pb = N.normalize_vector(N.as_vector(b.distribution))
        return float(np.clip(np.sum(np.sqrt(pa * pb)), 0.0, 1.0))  # Bhattacharyya 系数

    return 0.5


def _default_reliability_mapping(
    quality_part: dict[str, float],
    uncertainty_part: dict[str, float],
    consistency_mean: dict[str, float],
) -> dict[str, float]:
    """默认确定性映射：质量/确定性/一致性居中加权后 sigmoid。全 0.5 输入 → 0.5。"""
    w_q, w_u, w_c, gain = 0.4, 0.4, 0.2, 4.0
    out: dict[str, float] = {}
    for m in quality_part:
        raw = (
            w_q * (quality_part[m] - 0.5)
            + w_u * (uncertainty_part[m] - 0.5)
            + w_c * (consistency_mean.get(m, 0.5) - 0.5)
        )
        out[m] = N.sigmoid(gain * raw)
    return out


def compute_reliability_state(
    modality_outputs: Sequence[ModalityOutput],
    competition: Optional[CompetitionState] = None,
) -> ReliabilityState:
    """模态可靠性 R：质量(信号级) + 单模态确定性 + 跨模态一致性 → 每模态 ∈ [0,1]。

    仅对 ``available=True`` 的模态计算。``competition`` 给定时一致性走"竞争类别对方向"，
    否则走整分布重叠。最终聚合可经 ``register_reliability_head`` 替换为可学习头。
    """
    present = [o for o in modality_outputs if o.available]

    quality_part: dict[str, float] = {}
    uncertainty_part: dict[str, float] = {}
    for o in present:
        g = _quality_goodness(o.modality, o.quality)
        quality_part[o.modality] = 0.5 if g is None else g          # 无可测质量 → 中性先验
        c = _self_certainty(o)
        uncertainty_part[o.modality] = 0.0 if c is None else c       # 无分布（桩）→ 自certainty 0

    # 成对一致性（仅双方在场）。
    consistency_pairwise: dict[str, float] = {}
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            key = f"{a.modality}-{b.modality}"
            consistency_pairwise[key] = _pair_consistency(a, b, competition)

    # 每模态的平均一致性（参与的所有对）。
    consistency_mean: dict[str, float] = {}
    for o in present:
        vals = [
            v for k, v in consistency_pairwise.items() if o.modality in k.split("-")
        ]
        consistency_mean[o.modality] = float(sum(vals) / len(vals)) if vals else 0.5

    head = get_reliability_head()
    if head is not None:
        per_modality = head(dict(quality_part), dict(uncertainty_part), dict(consistency_pairwise))
        per_modality = {m: N.clip01(v) for m, v in per_modality.items()}
    else:
        per_modality = _default_reliability_mapping(quality_part, uncertainty_part, consistency_mean)

    return ReliabilityState(
        per_modality=per_modality,
        quality_part=quality_part,
        uncertainty_part=uncertainty_part,
        consistency_pairwise=consistency_pairwise,
    )
