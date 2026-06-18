"""模块3 三状态测试（论文创新点，确定性，逐项对照公式）。"""
from __future__ import annotations

import math

import pytest

from safemmsa import numerics as N
from safemmsa.contracts import ModalityOutput, ModalitySignalQuality
from safemmsa.labels import EMOTION_LABELS, NUM_CLASSES, index_of
from safemmsa.states import (
    compute_boundary_state,
    compute_competition_state,
    compute_reliability_state,
    register_reliability_head,
)


def _dist(**kw) -> dict[str, float]:
    """便捷构造分布（未给的类为 0），自动归一化。"""
    raw = {label: float(kw.get(label, 0.0)) for label in EMOTION_LABELS}
    return N.normalize_distribution(raw)


def _onehot(label: str) -> dict[str, float]:
    return {l: (1.0 if l == label else 0.0) for l in EMOTION_LABELS}


def _uniform() -> dict[str, float]:
    return {l: 1.0 / NUM_CLASSES for l in EMOTION_LABELS}


# ── 边界评分 B ─────────────────────────────────────────────────────
def test_boundary_onehot_is_zero():
    b = compute_boundary_state(_onehot("happy"))
    assert b.max_confidence == pytest.approx(1.0)
    assert b.top2_margin == pytest.approx(1.0)
    assert b.norm_entropy == pytest.approx(0.0, abs=1e-9)
    assert b.score == pytest.approx(0.0, abs=1e-9)


def test_boundary_uniform_is_near_max():
    b = compute_boundary_state(_uniform())
    # max_conf=1/7, margin=0, norm_entropy=1 → score = (1-1/7)/3 + 1/3 + 1/3
    expected = ((1 - 1 / 7) / 3) + (1 / 3) + (1 / 3)
    assert b.norm_entropy == pytest.approx(1.0)
    assert b.score == pytest.approx(min(1.0, expected))


def test_boundary_monotonic_in_uncertainty():
    sharp = compute_boundary_state(_dist(happy=0.9, sad=0.1))
    fuzzy = compute_boundary_state(_dist(happy=0.5, sad=0.45, angry=0.05))
    assert fuzzy.score > sharp.score


def test_boundary_weights_respected():
    p = _dist(happy=0.6, sad=0.3, angry=0.1)
    only_entropy = compute_boundary_state(p, weights=(0.0, 0.0, 1.0))
    assert only_entropy.score == pytest.approx(only_entropy.norm_entropy)


def test_boundary_score_bounded_unit_interval():
    for p in (_onehot("sad"), _uniform(), _dist(happy=0.4, fear=0.35, sad=0.25)):
        b = compute_boundary_state(p)
        assert 0.0 <= b.score <= 1.0


# ── 类别竞争 C ─────────────────────────────────────────────────────
def test_competition_top_pair_and_direction_sign():
    p0 = _dist(happy=0.5, sad=0.3, angry=0.2)
    # 文本强烈支持 happy 而非 sad；语音相反。
    logits_text = [0.0] * NUM_CLASSES
    logits_text[index_of("happy")] = 3.0
    logits_text[index_of("sad")] = -1.0
    logits_voice = [0.0] * NUM_CLASSES
    logits_voice[index_of("happy")] = -2.0
    logits_voice[index_of("sad")] = 2.0

    c = compute_competition_state(p0, {"text": logits_text, "audio": logits_voice})
    assert c.top1 == "happy" and c.top2 == "sad"
    assert c.per_modality_direction["text"] == pytest.approx(4.0)   # 3 - (-1)
    assert c.per_modality_direction["audio"] == pytest.approx(-4.0)  # -2 - 2
    assert c.margin == pytest.approx(p0["happy"] - p0["sad"])
    # 向量固定长度 5：video,audio,text 方向 + margin + entropy；缺 video → 0。
    assert len(c.vector) == 5
    assert c.vector[0] == 0.0  # video 缺席
    assert c.vector[1] == pytest.approx(-4.0)  # audio
    assert c.vector[2] == pytest.approx(4.0)   # text


def test_competition_handles_missing_logits():
    p0 = _dist(fear=0.4, surprise=0.35, neutral=0.25)
    c = compute_competition_state(p0, {})  # 无任何单模态 logits
    assert c.top1 == "fear" and c.top2 == "surprise"
    assert c.per_modality_direction == {}
    assert c.vector[:3] == [0.0, 0.0, 0.0]


# ── 模态可靠性 R ───────────────────────────────────────────────────
def _text_output(dist: dict[str, float], quality=None) -> ModalityOutput:
    logits = list(N.logits_from_distribution(dist))
    return ModalityOutput(
        modality="text",
        available=True,
        distribution=N.normalize_distribution(dist),
        logits=[float(x) for x in logits],
        quality=quality or ModalitySignalQuality(text_length=20, valid_token_ratio=1.0),
    )


def _av_stub(modality: str, quality=None) -> ModalityOutput:
    return ModalityOutput(
        modality=modality, available=True, distribution=None, logits=None,
        quality=quality or ModalitySignalQuality(),
    )


def test_reliability_only_available_modalities():
    out_text = _text_output(_dist(happy=0.8, sad=0.2))
    out_video = ModalityOutput(modality="video", available=False)
    r = compute_reliability_state([out_text, out_video])
    assert set(r.per_modality) == {"text"}
    assert 0.0 <= r.per_modality["text"] <= 1.0


def test_reliability_confident_modality_scores_higher():
    confident = _text_output(_dist(happy=0.95, sad=0.05))
    unsure = _text_output(_uniform())
    r_conf = compute_reliability_state([confident])
    r_unsure = compute_reliability_state([unsure])
    assert r_conf.per_modality["text"] > r_unsure.per_modality["text"]


def test_reliability_stub_av_has_zero_self_certainty():
    out = _av_stub("video")
    r = compute_reliability_state([out])
    assert r.uncertainty_part["video"] == pytest.approx(0.0)
    # 质量缺测 → 中性先验 0.5。
    assert r.quality_part["video"] == pytest.approx(0.5)


def test_reliability_pairwise_keys_and_range():
    a = _text_output(_dist(happy=0.8, sad=0.2))
    a.modality = "text"
    b = _av_stub("video")
    c = _av_stub("audio")
    r = compute_reliability_state([a, b, c])
    assert set(r.consistency_pairwise) == {"text-video", "text-audio", "video-audio"}
    for v in r.consistency_pairwise.values():
        assert 0.0 <= v <= 1.0


def test_reliability_head_override():
    out = _text_output(_dist(happy=0.8, sad=0.2))

    def head(q, u, c):
        return {m: 0.123 for m in q}

    register_reliability_head(head)
    r = compute_reliability_state([out])
    assert r.per_modality["text"] == pytest.approx(0.123)


def test_reliability_consistency_uses_competition_directions():
    # 两个文本风格模态在竞争对上方向相同 → 一致性高；相反 → 低。
    p0 = _dist(happy=0.5, sad=0.4, neutral=0.1)
    agree_a = _text_output(_dist(happy=0.9, sad=0.1))
    agree_b = _text_output(_dist(happy=0.85, sad=0.15))
    agree_b.modality = "audio"
    comp = compute_competition_state(p0, {})
    r = compute_reliability_state([agree_a, agree_b], competition=comp)
    assert r.consistency_pairwise["text-audio"] > 0.5
