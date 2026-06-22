"""模块11a — 持久化数据契约（pydantic v2），与 fig4-5 的关系型 E-R 一一对应。

这是 **JSON 持久化与后期 RDBMS 共用的同一套 schema 契约**：8 个规范化实体
（user / analysis_record / modality_output / state_diagnosis / modality_reliability /
eval_run / eval_item / operation_log），PK / FK 与 1-N、1-1 关系对齐 E-R 图。

compute 契约（``contracts.AnalyzeResult``）经 ``analyze_result_to_bundle`` 映射到这些表行；
``db/schema.sql`` 是同一 schema 的关系型落地，后期切到 RDBMS 即平滑迁移、对外契约零改动。

设计遵循 3NF：每个非键列只依赖其表主键；多值（各模态、各样本）拆为子表并以 FK 回指，
避免重复组与传递依赖。详见 docs/data-model.md。
"""
from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["user", "admin"]


def new_id(prefix: str) -> str:
    """带类型前缀的短主键，便于日志/调试里一眼看出表来源。"""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ── user 用户 ──────────────────────────────────────────────────────
class User(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(default_factory=lambda: new_id("usr"))  # PK
    username: str
    password_hash: str
    role: Role = "user"
    created_at: float = Field(default_factory=time.time)

    def public(self) -> dict:
        """不含 password_hash 的对外视图。"""
        return {"user_id": self.user_id, "username": self.username, "role": self.role}


# ── analysis_record 分析记录（1—N 来自 user）─────────────────────────
class AnalysisRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(default_factory=lambda: new_id("rec"))  # PK
    user_id: str                                                   # FK → user
    language: str
    final_emotion: str
    enhancement: str
    correction_magnitude: float
    bounded: bool = False
    initial_top: str                       # 初始预测 p⁰ 的 argmax 类
    boundary_score: float                  # 边界评分 B（冗余存一份，便于列表/排序）
    created_at: float = Field(default_factory=time.time)


# ── modality_output 模态输出（1—N 来自 analysis_record）──────────────
class ModalityOutputRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mo_id: str = Field(default_factory=lambda: new_id("mo"))  # PK
    record_id: str                                           # FK → analysis_record
    modality: str
    available: bool
    backend: str
    distribution: Optional[dict[str, float]] = None
    logits: Optional[list[float]] = None
    quality: dict[str, Optional[float]] = Field(default_factory=dict)  # ModalitySignalQuality 摊平


# ── state_diagnosis 状态诊断（1—1 与 analysis_record）────────────────
class StateDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sd_id: str = Field(default_factory=lambda: new_id("sd"))  # PK
    record_id: str                                           # FK → analysis_record（UNIQUE → 1:1）
    # 边界 B
    boundary_max_confidence: float
    boundary_top2_margin: float
    boundary_norm_entropy: float
    boundary_score: float
    # 类别竞争 C
    comp_top1: str
    comp_top2: str
    comp_margin: float
    comp_entropy: float
    comp_vector: list[float] = Field(default_factory=list)
    comp_per_modality_direction: dict[str, float] = Field(default_factory=dict)
    # 跨模态一致性（R 的记录级部分）
    consistency_pairwise: dict[str, float] = Field(default_factory=dict)


# ── modality_reliability 模态可靠性（1—N 来自 analysis_record）───────
class ModalityReliabilityRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mr_id: str = Field(default_factory=lambda: new_id("mr"))  # PK
    record_id: str                                           # FK → analysis_record
    modality: str
    reliability: float
    quality_part: Optional[float] = None
    uncertainty_part: Optional[float] = None


# ── 聚合视图：一条分析的完整快照（get_analysis 返回）─────────────────
class AnalysisBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record: AnalysisRecord
    modality_outputs: list[ModalityOutputRow] = Field(default_factory=list)
    state_diagnosis: StateDiagnosis
    reliabilities: list[ModalityReliabilityRow] = Field(default_factory=list)


# ── eval_run 评测运行（1—N 来自 user）────────────────────────────────
class EvalRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: new_id("run"))  # PK
    user_id: str                                               # FK → user
    dataset: Optional[str] = None
    n_items: int = 0
    macro_f1: Optional[float] = None
    accuracy: Optional[float] = None
    delta_macro_f1: Optional[float] = None
    delta_accuracy: Optional[float] = None
    mean_correction_magnitude: Optional[float] = None
    confusion: Optional[list[list[int]]] = None
    created_at: float = Field(default_factory=time.time)


# ── eval_item 评测样本（1—N 来自 eval_run）──────────────────────────
class EvalItemRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(default_factory=lambda: new_id("item"))  # PK
    run_id: str                                                  # FK → eval_run
    gold_label: str
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    text: Optional[str] = None
    pred_initial: Optional[str] = None
    pred_final: Optional[str] = None
    correction_magnitude: Optional[float] = None


# ── operation_log 操作日志（1—N 来自 user）──────────────────────────
class OperationLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_id: str = Field(default_factory=lambda: new_id("log"))  # PK
    user_id: Optional[str] = None                              # FK → user（系统事件可空）
    action: str
    detail: dict = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


# ── 映射：compute 契约 → 持久化表行 ────────────────────────────────
def analyze_result_to_bundle(
    result, *, user_id: str, record_id: Optional[str] = None
) -> AnalysisBundle:
    """把 ``AnalyzeResult``（compute 契约）拆成规范化的 AnalysisBundle（持久化契约）。

    纯函数、无 IO。JSON 后端与 SQLite 后端共用同一 bundle，落地差异只在写入层。
    """
    rid = record_id or new_id("rec")
    dist = result.initial.distribution
    initial_top = max(dist, key=dist.get) if dist else result.final_emotion

    record = AnalysisRecord(
        record_id=rid,
        user_id=user_id,
        language=result.language,
        final_emotion=result.final_emotion,
        enhancement=result.enhancement.enhancement,
        correction_magnitude=result.enhancement.correction_magnitude,
        bounded=result.enhancement.bounded,
        initial_top=initial_top,
        boundary_score=result.boundary.score,
    )

    modality_outputs = [
        ModalityOutputRow(
            record_id=rid,
            modality=m.modality,
            available=m.available,
            backend=m.backend,
            distribution=m.distribution,
            logits=m.logits,
            quality=m.quality.model_dump(),
        )
        for m in result.modalities
    ]

    c = result.competition
    b = result.boundary
    state_diagnosis = StateDiagnosis(
        record_id=rid,
        boundary_max_confidence=b.max_confidence,
        boundary_top2_margin=b.top2_margin,
        boundary_norm_entropy=b.norm_entropy,
        boundary_score=b.score,
        comp_top1=c.top1,
        comp_top2=c.top2,
        comp_margin=c.margin,
        comp_entropy=c.entropy,
        comp_vector=list(c.vector),
        comp_per_modality_direction=dict(c.per_modality_direction),
        consistency_pairwise=dict(result.reliability.consistency_pairwise),
    )

    rel = result.reliability
    reliabilities = [
        ModalityReliabilityRow(
            record_id=rid,
            modality=mod,
            reliability=score,
            quality_part=rel.quality_part.get(mod),
            uncertainty_part=rel.uncertainty_part.get(mod),
        )
        for mod, score in rel.per_modality.items()
    ]

    return AnalysisBundle(
        record=record,
        modality_outputs=modality_outputs,
        state_diagnosis=state_diagnosis,
        reliabilities=reliabilities,
    )
