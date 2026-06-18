"""模块7 — 编排器：predict-then-correct 全链路。

analyze(video?, audio?, text?) -> AnalyzeResult：
预处理质量 → 编码器(3) → 融合主干 → 三状态 → 有界增强 → 最终预测。

全程容错降级：任一模态缺失/低质/抛错都标 available=False，其余模态继续；
系统对外语种恒为 "mn"。即便全走桩，也能产出完整可观测的 AnalyzeResult。
"""
from __future__ import annotations

import time
from typing import Optional

from . import numerics as N
from .backbone import fuse
from .contracts import (
    MODALITIES,
    AnalyzeResult,
    Modality,
    ModalityOutput,
    ModalitySignalQuality,
    RawInput,
)
from .encoders import encode
from .enhancement import enhance
from .language import SYSTEM_LANGUAGE, detect_language
from .preprocessing import (
    extract_audio_quality,
    extract_text_quality,
    extract_video_quality,
)
from .states import (
    compute_boundary_state,
    compute_competition_state,
    compute_reliability_state,
)

_QUALITY_EXTRACTORS = {
    "video": lambda raw: extract_video_quality(raw.path),
    "audio": lambda raw: extract_audio_quality(raw.path),
    "text": lambda raw: extract_text_quality(raw.text),
}


def _process_modality(modality: Modality, raw: Optional[RawInput]) -> ModalityOutput:
    """单模态：质量提取 + 编码，任何异常都降级为 available=False。"""
    if raw is None:
        return ModalityOutput(modality=modality, available=False, backend="stub")
    try:
        quality = _QUALITY_EXTRACTORS[modality](raw)
    except Exception:
        quality = ModalitySignalQuality()
    try:
        return encode(modality, raw, quality)
    except Exception:
        return ModalityOutput(modality=modality, available=False, quality=quality, backend="stub")


def analyze(
    *,
    video: Optional[str] = None,
    audio: Optional[str] = None,
    text: Optional[str] = None,
    boundary_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> AnalyzeResult:
    """端到端分析。video/audio 为本地文件路径，text 为蒙古文（或任意）文本。"""
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    raws: dict[str, Optional[RawInput]] = {
        "video": RawInput(modality="video", path=video) if video else None,
        "audio": RawInput(modality="audio", path=audio) if audio else None,
        "text": RawInput(
            modality="text", text=text, language_hint=detect_language(text) if text else None
        )
        if text
        else None,
    }

    # 编码（含质量提取），始终产出 3 个模态条目（缺/失败→available=False）。
    t = time.perf_counter()
    outputs: list[ModalityOutput] = [
        _process_modality(m, raws[m]) for m in MODALITIES
    ]
    timings["encode_ms"] = round((time.perf_counter() - t) * 1000, 3)

    # 融合主干 → 初始预测 p⁰。
    t = time.perf_counter()
    initial = fuse(outputs)
    timings["backbone_ms"] = round((time.perf_counter() - t) * 1000, 3)

    # 三状态（确定性）。
    t = time.perf_counter()
    boundary = compute_boundary_state(initial.distribution, weights=boundary_weights)
    unimodal_logits = {
        o.modality: o.logits for o in outputs if o.available and o.logits is not None
    }
    competition = compute_competition_state(initial.distribution, unimodal_logits)
    reliability = compute_reliability_state(outputs, competition)
    timings["states_ms"] = round((time.perf_counter() - t) * 1000, 3)

    # 有界融合增强 → 最终预测 p*。
    t = time.perf_counter()
    enhancement = enhance(initial, boundary, competition, reliability)
    timings["enhancement_ms"] = round((time.perf_counter() - t) * 1000, 3)

    final_emotion = N.argmax_label(enhancement.final_distribution)
    timings["total_ms"] = round((time.perf_counter() - t_all) * 1000, 3)

    return AnalyzeResult(
        language=SYSTEM_LANGUAGE,
        modalities=outputs,
        initial=initial,
        boundary=boundary,
        competition=competition,
        reliability=reliability,
        enhancement=enhancement,
        final_emotion=final_emotion,
        timings_ms=timings,
    )
