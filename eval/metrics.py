"""指标：混淆矩阵 / accuracy / per-class P-R-F1 / macro-F1（纯 numpy，无 sklearn）。

混淆矩阵约定：行=真值，列=预测，标签顺序固定为 EMOTION_LABELS。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from safemmsa.labels import EMOTION_LABELS, NUM_CLASSES, index_of

from .schemas import EvalReport, PerClassMetric


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> np.ndarray:
    """KxK 整数混淆矩阵（行=真值，列=预测）。未知标签忽略并不计入。"""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t not in EMOTION_LABELS or p not in EMOTION_LABELS:
            continue
        cm[index_of(t), index_of(p)] += 1
    return cm


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """整体准确率。"""
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def _prf_from_cm(cm: np.ndarray) -> dict[str, PerClassMetric]:
    """从混淆矩阵算每类 precision/recall/f1/support。"""
    out: dict[str, PerClassMetric] = {}
    col_sum = cm.sum(axis=0)  # 预测为该类的总数
    row_sum = cm.sum(axis=1)  # 真值为该类的总数
    for i, label in enumerate(EMOTION_LABELS):
        tp = int(cm[i, i])
        precision = tp / col_sum[i] if col_sum[i] > 0 else 0.0
        recall = tp / row_sum[i] if row_sum[i] > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        out[label] = PerClassMetric(
            precision=round(float(precision), 6),
            recall=round(float(recall), 6),
            f1=round(float(f1), 6),
            support=int(row_sum[i]),
        )
    return out


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """macro-F1：对**出现过的真值类**（support>0）取 F1 平均（低资源不平衡更稳）。"""
    cm = confusion_matrix(y_true, y_pred)
    prf = _prf_from_cm(cm)
    present = [m.f1 for m in prf.values() if m.support > 0]
    if not present:
        return 0.0
    return round(float(np.mean(present)), 6)


def classification_report(y_true: Sequence[str], y_pred: Sequence[str]) -> EvalReport:
    """汇总成 EvalReport。"""
    cm = confusion_matrix(y_true, y_pred)
    prf = _prf_from_cm(cm)
    present = [m.f1 for m in prf.values() if m.support > 0]
    macro = round(float(np.mean(present)), 6) if present else 0.0
    return EvalReport(
        accuracy=round(accuracy(y_true, y_pred), 6),
        macro_f1=macro,
        per_class=prf,
        confusion=cm.tolist(),
        labels=list(EMOTION_LABELS),
        n=len(y_true),
    )
