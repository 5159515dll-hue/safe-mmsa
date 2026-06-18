"""模块4 — 模态编码器注册表（插点）+ 默认桩。

默认实现：
- 文本：关键词词典桩（zh/en；蒙语无词典 → 低置信中性，等真实 CINO 接入）。输出 7 类分布 + logits。
- 视频/音频：返回 ``available + quality``，但 ``distribution/logits/feature_ref=None``（无真实模型）。

训练完成后 ``register_encoder("text"/"video"/"audio", backend)`` 挂载真实
CINO / wav2vec2-XLSR-53 / ViT-FER 编码器（输出特征 + 单模态 logits + 分布），其余代码零改动。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from . import numerics as N
from .contracts import Modality, ModalityOutput, ModalitySignalQuality, RawInput
from .labels import EMOTION_LABELS, NUM_CLASSES
from .language import detect_language

# EncoderBackend(modality, raw_input, quality) -> ModalityOutput
EncoderBackend = Callable[[Modality, RawInput, ModalitySignalQuality], ModalityOutput]

_backends: dict[str, Optional[EncoderBackend]] = {"video": None, "audio": None, "text": None}


def register_encoder(modality: Modality, backend: Optional[EncoderBackend]) -> None:
    """注册某模态的真实编码器后端。传 None 恢复默认桩。"""
    if modality not in _backends:
        raise ValueError(f"unknown modality: {modality}")
    _backends[modality] = backend


def get_encoder(modality: Modality) -> Optional[EncoderBackend]:
    return _backends.get(modality)


def is_stub(modality: Modality) -> bool:
    """该模态当前是否在用默认桩（无真实模型）。供 /api/models/status 使用。"""
    return _backends.get(modality) is None


# ── 文本关键词桩（迁自 AIoT，扩为 7 类分布 + logits）─────────────────
_LEXICON: dict[str, dict[str, tuple[str, ...]]] = {
    "zh": {
        "happy": ("开心", "高兴", "快乐", "太好了", "幸福", "愉快", "喜欢", "满足", "棒", "好开心"),
        "sad": ("累", "难过", "伤心", "沮丧", "委屈", "想哭", "失落", "疲惫", "孤独", "难受", "压力"),
        "angry": ("生气", "愤怒", "烦", "讨厌", "气死", "恼火", "崩溃"),
        "fear": ("害怕", "担心", "紧张", "焦虑", "不安", "恐惧", "怕"),
        "surprise": ("惊讶", "没想到", "居然", "竟然", "震惊"),
        "disgust": ("恶心", "反感", "厌恶"),
    },
    "en": {
        "happy": ("happy", "glad", "great", "joy", "love", "wonderful", "awesome", "good"),
        "sad": ("tired", "sad", "down", "depressed", "lonely", "exhausted", "unhappy", "cry", "stressed"),
        "angry": ("angry", "mad", "annoyed", "furious", "hate"),
        "fear": ("afraid", "scared", "worried", "anxious", "nervous", "fear"),
        "surprise": ("surprised", "wow", "unexpected", "shocked"),
        "disgust": ("disgust", "gross", "sick"),
    },
}


def _stub_text_encoder(
    modality: Modality, raw: RawInput, quality: ModalitySignalQuality
) -> ModalityOutput:
    text = (raw.text or "").strip()
    if not text:
        return ModalityOutput(modality="text", available=False, quality=quality, backend="stub")

    language = raw.language_hint or detect_language(text)
    lowered = text.lower()
    scores = {label: 0.0 for label in EMOTION_LABELS}
    lexicon = _LEXICON.get(language, {})
    for emotion, words in lexicon.items():
        for word in words:
            if word in lowered:
                scores[emotion] += 1.0
    hits = sum(scores.values())

    uniform = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
    if hits <= 0:
        # 无命中（含蒙语无词典）→ 低置信中性，分布接近均匀但略偏 neutral。
        base = np.zeros(NUM_CLASSES)
        base[EMOTION_LABELS.index("neutral")] = 1.0
        conf = 0.2
    else:
        base = N.normalize_vector(N.as_vector(scores))
        conf = min(0.9, 0.5 + 0.1 * hits)

    dist = conf * base + (1.0 - conf) * uniform
    dist = N.normalize_vector(dist)
    logits = np.log(np.clip(dist, 1e-12, None))

    return ModalityOutput(
        modality="text",
        available=True,
        distribution=N.as_distribution(dist),
        logits=[float(x) for x in logits],
        feature_ref=None,
        quality=quality,
        backend="stub",
    )


def _stub_av_encoder(
    modality: Modality, raw: RawInput, quality: ModalitySignalQuality
) -> ModalityOutput:
    """视频/音频默认桩：有输入即 available，但无真实模型 → 无分布/logits/特征。"""
    available = raw.path is not None
    return ModalityOutput(
        modality=modality,
        available=available,
        distribution=None,
        logits=None,
        feature_ref=None,
        quality=quality,
        backend="stub",
    )


_DEFAULT_STUBS: dict[str, EncoderBackend] = {
    "text": _stub_text_encoder,
    "video": _stub_av_encoder,
    "audio": _stub_av_encoder,
}


def encode(modality: Modality, raw: RawInput, quality: ModalitySignalQuality) -> ModalityOutput:
    """对单模态编码：有注册后端用真实模型，否则用默认桩。"""
    backend = _backends.get(modality)
    if backend is not None:
        return backend(modality, raw, quality)
    return _DEFAULT_STUBS[modality](modality, raw, quality)
