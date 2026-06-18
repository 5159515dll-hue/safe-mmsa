"""模块1 — 固定 7 类情感标签与索引约定。

标签顺序是整套系统的**稳定契约**：所有 ``logits`` / ``list[float]`` 一律按
``EMOTION_LABELS`` 的顺序排列，训练产物只要遵守此顺序即可零改动接入。

注意：本系统的顺序为 (happy, sad, angry, fear, surprise, disgust, neutral)，
对齐 docs/system-design.md §3；与 AIoT Copilot 旧顺序（fear/surprise 互换）不同，
迁移时以本文件为准。
"""
from __future__ import annotations

from typing import Literal

# 固定 7 类（顺序即契约，勿改）。
EMOTION_LABELS: tuple[str, ...] = (
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "disgust",
    "neutral",
)

EmotionLabel = Literal["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]

NUM_CLASSES: int = len(EMOTION_LABELS)

# label -> 行内索引（logits/list 顺序）
LABEL_INDEX: dict[str, int] = {label: i for i, label in enumerate(EMOTION_LABELS)}

# 中文展示名（演示前端/报告用）。
LABEL_ZH: dict[str, str] = {
    "happy": "快乐",
    "sad": "悲伤",
    "angry": "愤怒",
    "fear": "恐惧",
    "surprise": "惊讶",
    "disgust": "厌恶",
    "neutral": "中性",
}


def index_of(label: str) -> int:
    """标签 -> 索引；非法标签抛 KeyError（调用方应只用 EMOTION_LABELS）。"""
    return LABEL_INDEX[label]


def label_at(index: int) -> str:
    """索引 -> 标签。"""
    return EMOTION_LABELS[index]
