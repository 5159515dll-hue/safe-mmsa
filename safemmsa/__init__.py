"""SAFE-MMSA 核心包（State-Aware Fusion-Enhanced Mongolian Multi-modal Sentiment Analysis）。

纯 Python / numpy / pydantic，**不依赖 torch**。对外稳定接口见 contracts.py；
端到端入口 ``analyze``；可训练部分（编码器/主干/增强/可靠性头）经 ``register_*`` 后插。
"""
from __future__ import annotations

from .backbone import register_fusion_backbone
from .contracts import (
    AnalyzeResult,
    BoundaryState,
    CompetitionState,
    EnhancementResult,
    InitialPrediction,
    Modality,
    ModalityOutput,
    ModalitySignalQuality,
    RawInput,
    ReliabilityState,
)
from .enhancement import (
    apply_bounded_correction,
    register_enhancement,
    register_reliability_head,
)
from .encoders import register_encoder
from .labels import EMOTION_LABELS, LABEL_ZH
from .language import SYSTEM_LANGUAGE
from .orchestrator import analyze
from .states import (
    compute_boundary_state,
    compute_competition_state,
    compute_reliability_state,
)

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "AnalyzeResult",
    "RawInput",
    "ModalityOutput",
    "ModalitySignalQuality",
    "InitialPrediction",
    "BoundaryState",
    "CompetitionState",
    "ReliabilityState",
    "EnhancementResult",
    "Modality",
    "EMOTION_LABELS",
    "LABEL_ZH",
    "SYSTEM_LANGUAGE",
    "compute_boundary_state",
    "compute_competition_state",
    "compute_reliability_state",
    "apply_bounded_correction",
    "register_encoder",
    "register_fusion_backbone",
    "register_enhancement",
    "register_reliability_head",
]
