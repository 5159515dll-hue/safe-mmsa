"""模块2 — 预处理与信号质量提取（纯信号级，不依赖情感模型 / torch）。

三个提取器各自把可选重依赖（opencv / librosa / transformers）做**惰性、容错**导入：
依赖缺失时对应质量项降级为 None，核心系统照常运行。
"""
from __future__ import annotations

from ..contracts import ModalitySignalQuality
from .audio_quality import extract_audio_quality
from .text_quality import extract_text_quality
from .video_quality import extract_video_quality

__all__ = [
    "ModalitySignalQuality",
    "extract_video_quality",
    "extract_audio_quality",
    "extract_text_quality",
]
