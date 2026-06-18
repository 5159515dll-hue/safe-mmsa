"""模块9 — 数据集适配 + 批量评测台（不依赖 torch）。

清单(JSONL) → runner → metrics(macro-F1/accuracy/混淆矩阵) → ablation/robustness → report。
对 initial(主干基线) 与 final(增强后) 分别出指标：默认 identity 时两者相等，框架就绪，
真模型注册后自然产生差异。
"""
from __future__ import annotations

from .metrics import accuracy, classification_report, confusion_matrix, macro_f1
from .runner import EvalItem, load_manifest, run_manifest
from .schemas import EvalReport, EvalRunResult, PairedReport

__all__ = [
    "EvalItem",
    "load_manifest",
    "run_manifest",
    "accuracy",
    "macro_f1",
    "confusion_matrix",
    "classification_report",
    "EvalReport",
    "PairedReport",
    "EvalRunResult",
]
