"""评测数据契约（pydantic v2）。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class PerClassMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precision: float
    recall: float
    f1: float
    support: int


class EvalReport(BaseModel):
    """单组预测（initial 或 final）的指标。"""

    model_config = ConfigDict(extra="forbid")

    accuracy: float
    macro_f1: float
    per_class: dict[str, PerClassMetric]
    confusion: list[list[int]]      # KxK，行=真值，列=预测，标签顺序=EMOTION_LABELS
    labels: list[str]
    n: int


class PairedReport(BaseModel):
    """initial(基线) 与 final(增强后) 并列指标 + 增量。"""

    model_config = ConfigDict(extra="forbid")

    initial: EvalReport
    final: EvalReport
    delta_macro_f1: float           # final − initial
    delta_accuracy: float
    mean_correction_magnitude: float
    n_changed: int                  # final 与 initial 预测不同的样本数


class EvalRunResult(BaseModel):
    """一次评测运行的完整结果。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    n_items: int
    paired: PairedReport
    per_item: Optional[list[dict]] = None   # 可选明细（id/label/initial/final/correction）
    config: dict
