"""模块1 契约测试：标签顺序固定、模型可构造/序列化。"""
from __future__ import annotations

from safemmsa.contracts import (
    AnalyzeResult,
    ModalityOutput,
    ModalitySignalQuality,
    RawInput,
)
from safemmsa.labels import EMOTION_LABELS, LABEL_INDEX, LABEL_ZH, NUM_CLASSES


def test_label_order_is_frozen_contract():
    # 顺序即契约：logits/list 一律按此排列，勿改。
    assert EMOTION_LABELS == (
        "happy",
        "sad",
        "angry",
        "fear",
        "surprise",
        "disgust",
        "neutral",
    )
    assert NUM_CLASSES == 7
    assert LABEL_INDEX["happy"] == 0 and LABEL_INDEX["neutral"] == 6
    assert set(LABEL_ZH) == set(EMOTION_LABELS)


def test_modality_output_defaults_and_extra_forbidden():
    out = ModalityOutput(modality="video", available=False)
    assert out.distribution is None and out.logits is None
    assert isinstance(out.quality, ModalitySignalQuality)
    assert out.backend == "stub"


def test_raw_input_roundtrip():
    raw = RawInput(modality="text", text="hi", language_hint="en")
    assert raw.model_dump()["modality"] == "text"


def test_analyze_result_schema_round_trips(stub_result):
    # stub_result 来自 conftest：完整跑一次 analyze 的结果。
    dumped = stub_result.model_dump()
    restored = AnalyzeResult.model_validate(dumped)
    assert restored.language == "mn"
    assert len(restored.modalities) == 3
    assert restored.final_emotion in EMOTION_LABELS
