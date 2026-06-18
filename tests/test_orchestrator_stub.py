"""模块5/7 端到端桩测：全走桩也能产出完整可观测的 AnalyzeResult；有界修正数学。"""
from __future__ import annotations

import numpy as np
import pytest

import safemmsa
from safemmsa import numerics as N
from safemmsa.contracts import EnhancementResult
from safemmsa.enhancement import DEFAULT_ETA, apply_bounded_correction, register_enhancement
from safemmsa.labels import EMOTION_LABELS, NUM_CLASSES, index_of


def test_analyze_text_only_end_to_end():
    r = safemmsa.analyze(text="I am so happy and great, wonderful awesome day")
    assert r.language == "mn"                       # 系统对外恒蒙古语
    assert len(r.modalities) == 3
    by_mod = {m.modality: m for m in r.modalities}
    assert by_mod["text"].available is True
    assert by_mod["video"].available is False
    assert by_mod["audio"].available is False
    assert r.final_emotion in EMOTION_LABELS
    assert r.final_emotion == "happy"               # 关键词桩应判 happy
    # 三状态齐备且类型正确。
    assert 0.0 <= r.boundary.score <= 1.0
    assert len(r.competition.vector) == 5
    assert set(r.reliability.per_modality) == {"text"}
    assert r.timings_ms is not None and "total_ms" in r.timings_ms


def test_analyze_chinese_sad():
    r = safemmsa.analyze(text="今天好难过 压力 好累 想哭")
    assert r.final_emotion == "sad"


def test_analyze_empty_is_neutral_fallback():
    r = safemmsa.analyze()
    assert all(m.available is False for m in r.modalities)
    assert r.final_emotion == "neutral"
    # 全缺 → 高不确定 → 边界评分偏高。
    assert r.boundary.score > 0.5


def test_identity_enhancement_keeps_initial():
    r = safemmsa.analyze(text="I love this wonderful happy day")
    assert r.enhancement.enhancement == "identity_baseline"
    assert r.enhancement.correction_magnitude == 0.0
    assert r.enhancement.final_distribution == r.initial.distribution


def test_backbone_and_initial_consistent():
    r = safemmsa.analyze(text="great great great wonderful")
    p0 = N.as_vector(r.initial.distribution)
    assert p0.sum() == pytest.approx(1.0)
    # z⁰ = log p⁰，softmax 还原应一致。
    recovered = N.softmax(r.initial.logits)
    assert np.allclose(recovered, p0, atol=1e-6)


# ── 有界修正数学（确定性） ─────────────────────────────────────────
def _uniform_dist():
    return {l: 1.0 / NUM_CLASSES for l in EMOTION_LABELS}


def _big_push_to(label: str):
    adj = [0.0] * NUM_CLASSES
    adj[index_of(label)] = 5.0
    return adj


def test_bounded_correction_zero_boundary_no_change():
    p_final, mag, bounded = apply_bounded_correction(
        _uniform_dist(), _big_push_to("happy"), boundary_score=0.0
    )
    assert mag == pytest.approx(0.0, abs=1e-9)
    assert bounded is True
    assert np.allclose(p_final, N.as_vector(_uniform_dist()), atol=1e-9)


def test_bounded_correction_respects_upper_bound():
    p_final, mag, bounded = apply_bounded_correction(
        _uniform_dist(), _big_push_to("happy"), boundary_score=1.0, eta=DEFAULT_ETA
    )
    assert mag <= DEFAULT_ETA + 1e-6
    assert bounded is True
    # 修正方向应朝 happy。
    assert p_final[index_of("happy")] > 1.0 / NUM_CLASSES


def test_bounded_correction_monotonic_in_boundary():
    mags = []
    for b in (0.1, 0.4, 0.8):
        _, mag, _ = apply_bounded_correction(_uniform_dist(), _big_push_to("sad"), boundary_score=b)
        mags.append(mag)
    assert mags[0] < mags[1] < mags[2]


def test_bounded_correction_small_adjustment_not_clipped():
    adj = [0.0] * NUM_CLASSES
    adj[index_of("fear")] = 0.05
    _, mag, bounded = apply_bounded_correction(_uniform_dist(), adj, boundary_score=1.0)
    assert bounded is False
    assert mag > 0.0


def test_registered_enhancement_changes_final():
    # 注册一个"用 B 门控、朝 top1 加强"的玩具增强后端，验证 final≠initial。
    def toy_backend(initial, boundary, competition, reliability) -> EnhancementResult:
        adj = [0.0] * NUM_CLASSES
        adj[index_of(competition.top1)] = 3.0
        p_final, mag, bounded = apply_bounded_correction(
            initial.distribution, adj, boundary.score
        )
        logits = np.log(np.clip(p_final, 1e-12, None))
        return EnhancementResult(
            final_distribution=N.as_distribution(p_final),
            final_logits=[float(x) for x in logits],
            correction_magnitude=mag,
            bounded=bounded,
            enhancement="trained_toy",
        )

    register_enhancement(toy_backend)
    r = safemmsa.analyze(text="happy sad angry fear")  # 制造一定竞争
    assert r.enhancement.enhancement == "trained_toy"
    # 当边界评分>0 时应有非零修正。
    if r.boundary.score > 0:
        assert r.enhancement.correction_magnitude > 0.0
