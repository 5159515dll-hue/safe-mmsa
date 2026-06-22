"""模块8/9 — 评测 API 路由：POST /api/eval/run、GET /api/eval/{id}。

支持两种清单来源：上传 JSONL 文件，或传服务器端清单路径（manifest_path）。
小数据集同步执行；结果落 persistence（eval_runs）并可按 run_id 取回。
"""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from safemmsa.persistence import eval_runs

from .runner import EvalItem, load_manifest, run_manifest
from .schemas import EvalRunResult

router = APIRouter(prefix="/api/eval", tags=["eval"])


def _items_from_upload(upload: UploadFile) -> list[EvalItem]:
    """把上传的 JSONL 落临时文件后解析（相对媒体路径将相对临时目录，建议用绝对路径）。"""
    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="safemmsa_manifest_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(upload.file.read())
        return load_manifest(path)
    finally:
        with suppress(OSError):
            os.remove(path)


@router.post("/run", response_model=EvalRunResult)
async def run_eval(
    manifest: Optional[UploadFile] = File(default=None),
    manifest_path: Optional[str] = Form(default=None),
    collect_per_item: bool = Form(default=True),
) -> EvalRunResult:
    """跑一次批量评测。清单二选一：上传 manifest 文件 或 传 manifest_path。"""
    if manifest is None and not manifest_path:
        raise HTTPException(status_code=400, detail="需提供 manifest 文件或 manifest_path")

    try:
        if manifest is not None and manifest.filename:
            items = _items_from_upload(manifest)
        else:
            if not os.path.exists(manifest_path or ""):
                raise HTTPException(status_code=404, detail=f"清单不存在: {manifest_path}")
            items = load_manifest(manifest_path)  # type: ignore[arg-type]
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"清单 JSON 解析失败: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not items:
        raise HTTPException(status_code=400, detail="清单为空")

    result = run_manifest(items, collect_per_item=True)
    eval_runs.append(result.model_dump(), record_id=result.run_id)
    # 同时落规范化的 eval_run / eval_item（与 fig4-5 的 E-R 对齐）；失败不影响返回。
    with suppress(Exception):
        _persist_eval_run(result, items)
    if not collect_per_item:
        result.per_item = None
    return result


def _persist_eval_run(result: EvalRunResult, items: list[EvalItem]) -> None:
    from safemmsa.records import EvalItemRow, EvalRun
    from safemmsa.repository import ANONYMOUS_USER_ID, get_repository

    paired = result.paired
    run = EvalRun(
        run_id=result.run_id,
        user_id=ANONYMOUS_USER_ID,
        dataset=str(result.config.get("dataset")) if result.config.get("dataset") else None,
        n_items=result.n_items,
        macro_f1=paired.final.macro_f1,
        accuracy=paired.final.accuracy,
        delta_macro_f1=paired.delta_macro_f1,
        delta_accuracy=paired.delta_accuracy,
        mean_correction_magnitude=paired.mean_correction_magnitude,
        confusion=paired.final.confusion,
    )
    preds = {p["id"]: p for p in (result.per_item or [])}
    rows = [
        EvalItemRow(
            run_id=result.run_id,
            gold_label=it.label,
            video_path=it.video_path,
            audio_path=it.audio_path,
            text=it.text,
            pred_initial=preds.get(it.id, {}).get("initial"),
            pred_final=preds.get(it.id, {}).get("final"),
            correction_magnitude=preds.get(it.id, {}).get("correction_magnitude"),
        )
        for it in items
    ]
    get_repository().save_eval_run(run, rows)


@router.get("/{run_id}", response_model=EvalRunResult)
def get_eval(run_id: str) -> EvalRunResult:
    """按 run_id 取回评测结果。"""
    record = eval_runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"找不到评测 run: {run_id}")
    # 去掉 store 注入的 id/created_at，再按契约校验返回。
    record = {k: v for k, v in record.items() if k not in ("id", "created_at")}
    return EvalRunResult.model_validate(record)


@router.get("")
def list_eval(limit: int = 20) -> dict:
    """列出最近的评测 run（摘要）。"""
    runs = eval_runs.list(limit=limit)
    return {
        "runs": [
            {
                "run_id": r.get("run_id"),
                "n_items": r.get("n_items"),
                "created_at": r.get("created_at"),
                "final_macro_f1": r.get("paired", {}).get("final", {}).get("macro_f1"),
                "delta_macro_f1": r.get("paired", {}).get("delta_macro_f1"),
            }
            for r in runs
        ]
    }
