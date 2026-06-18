"""模块1 — 核心数据契约（pydantic v2），对齐 docs/system-design.md §3。

这是整套系统的**稳定接口**：编码器/融合主干/增强网络等训练产物只要满足这些签名，
即可经 ``register_*`` 零改动接入。本文件不依赖 torch。
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .labels import EMOTION_LABELS  # noqa: F401  (re-export 方便外部 import)

Modality = Literal["video", "audio", "text"]
MODALITIES: tuple[Modality, ...] = ("video", "audio", "text")


# ── 原始输入容器（编码器后端入参）─────────────────────────────────
class RawInput(BaseModel):
    """单模态原始输入引用：文件路径（视频/音频）或文本字符串。

    核心包不在内存里搬运大文件——视频/音频以本地路径传递，编码器/质量提取器自行读取。
    """

    model_config = ConfigDict(extra="forbid")

    modality: Modality
    path: Optional[str] = None          # video/audio：本地文件路径
    text: Optional[str] = None          # text：原文
    language_hint: Optional[str] = None # detect_language 的提示（不改变系统对外的 "mn"）


# ── 信号质量（与情感弱相关、不依赖深度模型；缺测留 None）───────────
class ModalitySignalQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 视频
    face_detection_rate: Optional[float] = None
    blur: Optional[float] = None
    brightness: Optional[float] = None
    frame_loss_rate: Optional[float] = None
    # 音频
    snr_db: Optional[float] = None
    voiced_ratio: Optional[float] = None
    silence_ratio: Optional[float] = None
    # 文本
    text_length: Optional[int] = None
    valid_token_ratio: Optional[float] = None
    low_freq_subword_ratio: Optional[float] = None


# ── 单模态编码器输出 ───────────────────────────────────────────────
class ModalityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modality: Modality
    available: bool
    distribution: Optional[dict[str, float]] = None  # 单模态 head softmax（7 类）
    logits: Optional[list[float]] = None             # 单模态 logits（状态计算需要）
    feature_ref: Optional[str] = None                # 高层特征缓存键（融合主干用；桩为 None）
    quality: ModalitySignalQuality = Field(default_factory=ModalitySignalQuality)
    backend: str = "stub"                            # "stub" | "trained_<name>"


# ── 初始预测（融合主干输出）────────────────────────────────────────
class InitialPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distribution: dict[str, float]              # p⁰
    logits: list[float]                         # z⁰
    fused_feature_ref: Optional[str] = None     # h⁰ 引用
    backbone: str                               # "late_weighted_baseline" | "trained_<name>"


# ── 三状态（论文创新点，确定性计算）────────────────────────────────
class BoundaryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_confidence: float
    top2_margin: float
    norm_entropy: float
    score: float                                # B ∈ [0,1]，门控修正幅度


class CompetitionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top1: str
    top2: str
    per_modality_direction: dict[str, float]    # 各模态对 (top1,top2) 的方向支持；>0 偏向 top1
    margin: float
    entropy: float
    vector: list[float]                         # C 状态向量（供增强网络 D 输入）


class ReliabilityState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_modality: dict[str, float]              # R：每模态可靠性 ∈ [0,1]
    quality_part: dict[str, float]
    uncertainty_part: dict[str, float]
    consistency_pairwise: dict[str, float]      # {"video-audio":.., "video-text":.., "audio-text":..}


# ── 融合增强结果（最终预测）────────────────────────────────────────
class EnhancementResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_distribution: dict[str, float]        # p*
    final_logits: list[float]
    correction_magnitude: float                 # ‖p*−p⁰‖，受 B 上界约束
    bounded: bool                               # 是否触发了上界裁剪
    enhancement: str                            # "identity_baseline" | "trained_<name>"


# ── 端到端结果 ─────────────────────────────────────────────────────
class AnalyzeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str                               # 固定 "mn"
    modalities: list[ModalityOutput]
    initial: InitialPrediction
    boundary: BoundaryState
    competition: CompetitionState
    reliability: ReliabilityState
    enhancement: EnhancementResult
    final_emotion: str
    timings_ms: Optional[dict[str, float]] = None
