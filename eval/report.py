"""导出评测结果：JSON / CSV + 混淆矩阵图（matplotlib 可选）。"""
from __future__ import annotations

import csv
import json
import os
from typing import Optional

from .schemas import EvalReport, EvalRunResult


def to_json(result: EvalRunResult, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    return path


def _write_per_class_csv(report: EvalReport, path: str, prefix: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"{prefix}_label", "precision", "recall", "f1", "support"])
        for label, m in report.per_class.items():
            w.writerow([label, m.precision, m.recall, m.f1, m.support])
        w.writerow([])
        w.writerow([f"{prefix}_accuracy", report.accuracy])
        w.writerow([f"{prefix}_macro_f1", report.macro_f1])


def to_csv(result: EvalRunResult, path: str) -> str:
    """把 initial 与 final 的每类指标写入一个 CSV。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", result.run_id, "n", result.n_items])
        w.writerow(["set", "label", "precision", "recall", "f1", "support"])
        for set_name, rep in (("initial", result.paired.initial), ("final", result.paired.final)):
            for label, m in rep.per_class.items():
                w.writerow([set_name, label, m.precision, m.recall, m.f1, m.support])
        w.writerow([])
        w.writerow(["summary", "accuracy", "macro_f1"])
        w.writerow(["initial", result.paired.initial.accuracy, result.paired.initial.macro_f1])
        w.writerow(["final", result.paired.final.accuracy, result.paired.final.macro_f1])
        w.writerow(["delta", result.paired.delta_accuracy, result.paired.delta_macro_f1])
    return path


def confusion_png(report: EvalReport, path: str, title: str = "Confusion Matrix") -> Optional[str]:
    """画混淆矩阵热图（论文可直接用）。无 matplotlib → 返回 None。"""
    try:  # pragma: no cover - 依赖环境
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None

    cm = np.array(report.confusion)
    labels = report.labels
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def export(result: EvalRunResult, outdir: str) -> dict[str, Optional[str]]:
    """一次性导出 JSON + CSV + 两张混淆矩阵图，返回写出的路径。"""
    os.makedirs(outdir, exist_ok=True)
    paths: dict[str, Optional[str]] = {
        "json": to_json(result, os.path.join(outdir, f"{result.run_id}.json")),
        "csv": to_csv(result, os.path.join(outdir, f"{result.run_id}.csv")),
        "confusion_initial_png": confusion_png(
            result.paired.initial,
            os.path.join(outdir, f"{result.run_id}_initial_cm.png"),
            title="Initial (baseline)",
        ),
        "confusion_final_png": confusion_png(
            result.paired.final,
            os.path.join(outdir, f"{result.run_id}_final_cm.png"),
            title="Final (enhanced)",
        ),
    }
    return paths
