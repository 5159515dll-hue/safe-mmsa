"""清单加载 + 批量运行：遍历 → analyze → 收集 initial 与 final → 配对指标。"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import safemmsa
from safemmsa import numerics as N

from .metrics import classification_report
from .schemas import EvalRunResult, PairedReport


@dataclass
class EvalItem:
    """清单中的一条样本。"""

    id: str
    label: str
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    text: Optional[str] = None


# 对样本做扰动（鲁棒性注入用）：输入一条 → 输出一条（可改路径/文本/置空模态）。
Transform = Callable[[EvalItem], EvalItem]


def load_manifest(path: str) -> list[EvalItem]:
    """读 JSONL 清单。相对的 video/audio 路径按清单所在目录解析。"""
    base = os.path.dirname(os.path.abspath(path))
    items: list[EvalItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj or "label" not in obj:
                raise ValueError(f"清单第 {lineno} 行缺少 id/label")

            def _resolve(p: Optional[str]) -> Optional[str]:
                if not p:
                    return None
                return p if os.path.isabs(p) else os.path.join(base, p)

            items.append(
                EvalItem(
                    id=str(obj["id"]),
                    label=str(obj["label"]),
                    video_path=_resolve(obj.get("video_path")),
                    audio_path=_resolve(obj.get("audio_path")),
                    text=obj.get("text"),
                )
            )
    return items


def run_manifest(
    items: Sequence[EvalItem],
    *,
    transform: Optional[Transform] = None,
    collect_per_item: bool = False,
    run_id: Optional[str] = None,
    config: Optional[dict] = None,
) -> EvalRunResult:
    """对每条样本跑 analyze，收集 initial/final 预测，产出配对指标。"""
    run_id = run_id or uuid.uuid4().hex[:12]
    labels: list[str] = []
    initial_preds: list[str] = []
    final_preds: list[str] = []
    corrections: list[float] = []
    per_item: list[dict] = []

    for item in items:
        it = transform(item) if transform else item
        result = safemmsa.analyze(
            video=it.video_path, audio=it.audio_path, text=it.text
        )
        initial_pred = N.argmax_label(result.initial.distribution)
        final_pred = result.final_emotion
        corr = result.enhancement.correction_magnitude

        labels.append(item.label)
        initial_preds.append(initial_pred)
        final_preds.append(final_pred)
        corrections.append(corr)
        if collect_per_item:
            per_item.append(
                {
                    "id": item.id,
                    "label": item.label,
                    "initial": initial_pred,
                    "final": final_pred,
                    "correction_magnitude": corr,
                    "changed": initial_pred != final_pred,
                }
            )

    initial_report = classification_report(labels, initial_preds)
    final_report = classification_report(labels, final_preds)
    n_changed = sum(1 for a, b in zip(initial_preds, final_preds) if a != b)
    mean_corr = round(sum(corrections) / len(corrections), 6) if corrections else 0.0

    paired = PairedReport(
        initial=initial_report,
        final=final_report,
        delta_macro_f1=round(final_report.macro_f1 - initial_report.macro_f1, 6),
        delta_accuracy=round(final_report.accuracy - initial_report.accuracy, 6),
        mean_correction_magnitude=mean_corr,
        n_changed=n_changed,
    )
    return EvalRunResult(
        run_id=run_id,
        n_items=len(items),
        paired=paired,
        per_item=per_item if collect_per_item else None,
        config=config or {},
    )
