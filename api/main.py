"""模块8 — FastAPI 应用：文件上传 → 端到端分析 + 健康检查 + 模型状态（+ 评测路由）。

后端只绑 127.0.0.1（部署时由 nginx 反代）。三模态经 multipart 进入：
video/audio 作为文件上传落临时文件，text 作为表单字段。
"""
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager, suppress
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import safemmsa
from safemmsa.contracts import AnalyzeResult
from safemmsa.persistence import analyze_history

from .models_status import models_status

# 单文件大小上限（部署机内存有限，防 OOM）。
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """启动时按 ENV(SAFEMMSA_MODELS_CONFIG) 挂载真实模型；无配置则全用桩/基线。"""
    with suppress(Exception):
        from models import load_models  # type: ignore

        load_models()
    yield


app = FastAPI(
    title="SAFE-MMSA API",
    version=safemmsa.__version__,
    description="状态感知驱动融合增强的蒙古语多模态情感分析（predict-then-correct）",
    lifespan=_lifespan,
)

# 允许前端跨域（Next.js 演示页）。可用 SAFEMMSA_CORS_ORIGINS 收紧。
_origins = os.environ.get("SAFEMMSA_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 评测路由（模块9）；eval 包就绪时挂载，缺失不影响核心 API。
with suppress(Exception):
    from eval.routes import router as eval_router  # type: ignore

    app.include_router(eval_router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": safemmsa.__version__, "language": safemmsa.SYSTEM_LANGUAGE}


@app.get("/api/models/status")
def models_status_endpoint() -> dict:
    """各 register_* 插点的桩/真模型状态。"""
    return models_status()


@app.get("/api/history")
def history(limit: int = 20) -> dict:
    """最近的分析历史（紧凑摘要）。"""
    return {"items": analyze_history.list(limit=limit)}


def _spill_to_temp(upload: Optional[UploadFile], suffix: str) -> Optional[str]:
    """把上传文件落到临时路径（带大小上限）；无文件返回 None。调用方负责删除。"""
    if upload is None or not upload.filename:
        return None
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="safemmsa_")
    written = 0
    with os.fdopen(fd, "wb") as out:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > _MAX_UPLOAD_BYTES:
                out.close()
                with suppress(OSError):
                    os.remove(path)
                raise ValueError(f"上传文件超过上限 {_MAX_UPLOAD_BYTES // (1024*1024)}MB")
            out.write(chunk)
    if written == 0:
        with suppress(OSError):
            os.remove(path)
        return None
    return path


@app.post("/api/analyze", response_model=AnalyzeResult)
async def analyze(
    text: Optional[str] = Form(default=None),
    video: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
) -> AnalyzeResult:
    """三模态（任意子集）→ 完整 AnalyzeResult：初始预测 → 三状态 → 最终 + 修正幅度。"""
    video_path = _spill_to_temp(video, ".mp4")
    audio_path = _spill_to_temp(audio, ".wav")
    try:
        result = safemmsa.analyze(
            video=video_path,
            audio=audio_path,
            text=(text.strip() if text and text.strip() else None),
        )
        # 持久化一条紧凑历史（不含原始媒体/文本，仅结果摘要）。
        with suppress(Exception):
            analyze_history.append(
                {
                    "final_emotion": result.final_emotion,
                    "language": result.language,
                    "initial_top": max(result.initial.distribution, key=result.initial.distribution.get),
                    "boundary_score": round(result.boundary.score, 4),
                    "correction_magnitude": result.enhancement.correction_magnitude,
                    "available": {m.modality: m.available for m in result.modalities},
                }
            )
        return result
    finally:
        for p in (video_path, audio_path):
            if p:
                with suppress(OSError):
                    os.remove(p)


# 运行：uvicorn api.main:app --host 127.0.0.1 --port 8100
